import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type Model,
	normalizeContext,
	type Usage,
} from "@earendil-works/pi-ai";
import { getBuiltinModel } from "@earendil-works/pi-ai/providers/all";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CacheWarmer, type CacheWarmRequest } from "../../src/core/cache-warmer.ts";
import type { SessionManager } from "../../src/core/session-manager.ts";
import { createHarness, type Harness } from "./harness.ts";

const model: Model<Api> = {
	...getBuiltinModel("anthropic", "claude-opus-4-6"),
	promptCache: { short: 300, long: 3600 },
};

const request: CacheWarmRequest = {
	model,
	context: normalizeContext({ messages: [] }),
	options: {},
};

const current = () => true;

function usage(tokens: number): Usage {
	return {
		input: tokens,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: tokens,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function assistantMessage(model: Model<Api>, text: string, tokens: number): AssistantMessage {
	return {
		...fauxAssistantMessage(text),
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: usage(tokens),
	};
}

function createWarmer(): { warmer: CacheWarmer; calls: Model<Api>[]; sessionManager: { current?: SessionManager } } {
	const calls: Model<Api>[] = [];
	const sessionManager: { current?: SessionManager } = {};
	const warmer = new CacheWarmer(
		{
			streamSimple: (model) => {
				calls.push(model);
				return {
					result: async () => assistantMessage(model, "", 1),
				} as unknown as AssistantMessageEventStream;
			},
		},
		{
			appendUsage: (kind, provider, modelId, usage, note) =>
				sessionManager.current!.appendUsage(kind, provider, modelId, usage, note),
			getBranch: () => sessionManager.current!.getBranch(),
		},
		() => "idle",
		async (event) => event.action,
	);
	return { warmer, calls, sessionManager };
}

function seedSession(harness: Harness, tokens: number): void {
	const now = Date.now();
	harness.sessionManager.appendMessage({
		role: "user",
		content: [{ type: "text", text: "message to compact" }],
		timestamp: now - 1000,
	});
	harness.sessionManager.appendMessage(assistantMessage(model, "assistant response to compact", tokens));
	harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;
}

function respondWithSummary(harness: Harness): void {
	harness.session.agent.streamFunction = (model) => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() => {
			stream.push({
				type: "done",
				reason: "stop",
				message: assistantMessage(model, "idle summary", 10),
			});
		});
		return stream;
	};
}

describe("AgentSession idle compaction", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("compacts once at the one-hour idle limit and hands off from warming", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(0);
		const { warmer, calls, sessionManager } = createWarmer();
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 1_000_000, maxTokens: 100 }],
			cacheWarmer: warmer,
			settings: { compaction: { keepRecentTokens: 1, idle: { enabled: true } } },
		});
		harnesses.push(harness);
		sessionManager.current = harness.sessionManager;
		seedSession(harness, 200_000);
		respondWithSummary(harness);

		const compactionEnd = new Promise<void>((resolve) => {
			harness.session.subscribe((event) => {
				if (event.type === "compaction_end" && event.reason === "idle") resolve();
			});
		});
		warmer.start(request, current);
		warmer.onAgentSettled();
		await vi.advanceTimersByTimeAsync(3_600_000);
		await compactionEnd;

		expect(calls).toHaveLength(13);
		expect(harness.eventsOfType("compaction_start").filter((event) => event.reason === "idle")).toHaveLength(1);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({ reason: "idle", aborted: false });
		expect(harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(1);

		await vi.advanceTimersByTimeAsync(3_600_000);
		expect(calls).toHaveLength(13);
		expect(harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(1);
	});

	it("neither warms nor compacts a session below the idle threshold", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(0);
		const { warmer, calls, sessionManager } = createWarmer();
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 1_000_000, maxTokens: 100 }],
			cacheWarmer: warmer,
			settings: { compaction: { keepRecentTokens: 1, idle: { enabled: true } } },
		});
		harnesses.push(harness);
		sessionManager.current = harness.sessionManager;
		seedSession(harness, 99_999);

		expect(warmer.isIdleWarmingEligible?.()).toBe(false);
		warmer.start(request, current);
		warmer.onAgentSettled();
		await vi.advanceTimersByTimeAsync(3_600_000);

		expect(warmer.status).toMatchObject({ state: "inactive", reason: "context below idle warming threshold" });
		expect(calls).toHaveLength(0);
		expect(harness.eventsOfType("compaction_start")).toHaveLength(0);
		expect(harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(0);
	});

	it("treats the idle threshold as inclusive", async () => {
		const { warmer, sessionManager } = createWarmer();
		const harness = await createHarness({
			cacheWarmer: warmer,
			settings: { compaction: { idle: { enabled: true } } },
		});
		harnesses.push(harness);
		sessionManager.current = harness.sessionManager;
		seedSession(harness, 100_000);

		expect(warmer.isIdleWarmingEligible?.()).toBe(true);
	});

	it("cancels the handoff when the user returns before the idle limit", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(0);
		const { warmer, sessionManager } = createWarmer();
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 1_000_000, maxTokens: 100 }],
			cacheWarmer: warmer,
			settings: { compaction: { keepRecentTokens: 1, idle: { enabled: true } } },
		});
		harnesses.push(harness);
		sessionManager.current = harness.sessionManager;
		seedSession(harness, 200_000);
		harness.setResponses([fauxAssistantMessage("back at the keyboard")]);

		warmer.start(request, current);
		warmer.onAgentSettled();
		await vi.advanceTimersByTimeAsync(3_540_000);

		const returned = harness.session.prompt("I am back");
		warmer.start(request, current);
		await returned;
		await vi.advanceTimersByTimeAsync(120_000);

		expect(harness.eventsOfType("compaction_start").filter((event) => event.reason === "idle")).toHaveLength(0);
		expect(harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(0);
	});

	it("does not compact when idle compaction is disabled", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(0);
		const { warmer, sessionManager } = createWarmer();
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 1_000_000, maxTokens: 100 }],
			cacheWarmer: warmer,
			settings: { compaction: { keepRecentTokens: 1, idle: { enabled: false } } },
		});
		harnesses.push(harness);
		sessionManager.current = harness.sessionManager;
		seedSession(harness, 200_000);
		respondWithSummary(harness);

		warmer.start(request, current);
		warmer.onAgentSettled();
		await vi.advanceTimersByTimeAsync(3_600_000);

		expect(warmer.status).toMatchObject({ state: "inactive", reason: "one-hour idle limit reached" });
		expect(harness.eventsOfType("compaction_start").filter((event) => event.reason === "idle")).toHaveLength(0);
		expect(harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(0);
	});
});
