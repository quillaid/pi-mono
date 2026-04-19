import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { type BedrockOptions, streamBedrock } from "../src/providers/amazon-bedrock.js";
import type { Context, Model } from "../src/types.js";

interface BedrockThinkingPayload {
	modelId?: string;
	additionalModelRequestFields?: {
		thinking?: { type: string; budget_tokens?: number; display?: string };
		output_config?: { effort?: string };
		anthropic_beta?: string[];
	};
}

function makeContext(): Context {
	return {
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};
}

async function capturePayload(
	model: Model<"bedrock-converse-stream">,
	options?: BedrockOptions,
): Promise<BedrockThinkingPayload> {
	let capturedPayload: BedrockThinkingPayload | undefined;
	const s = streamBedrock(model, makeContext(), {
		...options,
		reasoning: options?.reasoning ?? "high",
		signal: AbortSignal.abort(),
		onPayload: (payload) => {
			capturedPayload = payload as BedrockThinkingPayload;
			return payload;
		},
	});

	for await (const event of s) {
		if (event.type === "error") {
			break;
		}
	}

	if (!capturedPayload) {
		throw new Error("Expected Bedrock payload to be captured before request abort");
	}

	return capturedPayload;
}

describe("Bedrock thinking payload", () => {
	it("uses adaptive thinking for Claude Opus 4.7 when reasoning is enabled", async () => {
		const baseModel = getModel("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "global.anthropic.claude-opus-4-7-v1",
			name: "Claude Opus 4.7 (Global)",
		};

		const payload = await capturePayload(model);

		expect(payload.additionalModelRequestFields?.thinking).toEqual({ type: "adaptive", display: "summarized" });
		expect(payload.additionalModelRequestFields?.output_config).toEqual({ effort: "high" });
		expect(payload.additionalModelRequestFields?.anthropic_beta).toBeUndefined();
	});

	it("maps xhigh reasoning to effort=xhigh for Claude Opus 4.7", async () => {
		const baseModel = getModel("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "global.anthropic.claude-opus-4-7-v1",
			name: "Claude Opus 4.7 (Global)",
		};

		const payload = await capturePayload(model, { reasoning: "xhigh" });

		expect(payload.additionalModelRequestFields?.thinking).toEqual({ type: "adaptive", display: "summarized" });
		expect(payload.additionalModelRequestFields?.output_config).toEqual({ effort: "xhigh" });
		expect(payload.additionalModelRequestFields?.anthropic_beta).toBeUndefined();
	});

	it("omits display for GovCloud model ids on non-adaptive Claude thinking", async () => {
		const baseModel = getModel("amazon-bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
			name: "Claude Sonnet 4.5 (GovCloud)",
		};

		const payload = await capturePayload(model);

		expect(payload.additionalModelRequestFields?.thinking).toEqual({ type: "enabled", budget_tokens: 16384 });
		expect(payload.additionalModelRequestFields?.anthropic_beta).toEqual(["interleaved-thinking-2025-05-14"]);
	});

	it("omits display for GovCloud regions on adaptive Claude thinking", async () => {
		const baseModel = getModel("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "global.anthropic.claude-opus-4-7-v1",
			name: "Claude Opus 4.7 (Global)",
		};

		const payload = await capturePayload(model, { region: "us-gov-west-1" });

		expect(payload.additionalModelRequestFields?.thinking).toEqual({ type: "adaptive" });
		expect(payload.additionalModelRequestFields?.output_config).toEqual({ effort: "high" });
		expect(payload.additionalModelRequestFields?.anthropic_beta).toBeUndefined();
	});
});

describe("Bedrock inference profile resolution", () => {
	it("auto-prefixes bare Opus 4.6 model ID with us.", async () => {
		const baseModel = getModel("amazon-bedrock", "us.anthropic.claude-opus-4-6-v1");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "anthropic.claude-opus-4-6-v1",
		};

		const payload = await capturePayload(model);
		expect(payload.modelId).toBe("us.anthropic.claude-opus-4-6-v1");
	});

	it("auto-prefixes bare Opus 4.7 model ID with us.", async () => {
		const baseModel = getModel("amazon-bedrock", "us.anthropic.claude-opus-4-7");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "anthropic.claude-opus-4-7",
		};

		const payload = await capturePayload(model);
		expect(payload.modelId).toBe("us.anthropic.claude-opus-4-7");
	});

	it("auto-prefixes bare Opus 4.5 model ID with us.", async () => {
		const baseModel = getModel("amazon-bedrock", "us.anthropic.claude-opus-4-5-20251101-v1:0");
		const model: Model<"bedrock-converse-stream"> = {
			...baseModel,
			id: "anthropic.claude-opus-4-5-20251101-v1:0",
		};

		const payload = await capturePayload(model);
		expect(payload.modelId).toBe("us.anthropic.claude-opus-4-5-20251101-v1:0");
	});

	it("preserves us. prefixed model IDs", async () => {
		const model = getModel("amazon-bedrock", "us.anthropic.claude-opus-4-7");

		const payload = await capturePayload(model);
		expect(payload.modelId).toBe("us.anthropic.claude-opus-4-7");
	});

	it("preserves eu. prefixed model IDs", async () => {
		const model = getModel("amazon-bedrock", "eu.anthropic.claude-opus-4-7");

		const payload = await capturePayload(model);
		expect(payload.modelId).toBe("eu.anthropic.claude-opus-4-7");
	});

	it("preserves global. prefixed model IDs", async () => {
		const model = getModel("amazon-bedrock", "global.anthropic.claude-opus-4-7");

		const payload = await capturePayload(model);
		expect(payload.modelId).toBe("global.anthropic.claude-opus-4-7");
	});

	it("does not prefix older Claude models that work with on-demand", async () => {
		const model = getModel("amazon-bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0");

		const payload = await capturePayload(model, { reasoning: undefined });
		expect(payload.modelId).toBe("anthropic.claude-3-5-haiku-20241022-v1:0");
	});

	it("does not prefix Nova models", async () => {
		const model = getModel("amazon-bedrock", "amazon.nova-lite-v1:0");

		const payload = await capturePayload(model, { reasoning: undefined });
		expect(payload.modelId).toBe("amazon.nova-lite-v1:0");
	});
});
