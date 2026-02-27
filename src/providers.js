import Anthropic from "@anthropic-ai/sdk";
import { StudentProvider } from "./student.js";

// --- Shared helpers ---

function parseJSON(text) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`No JSON found in response:\n${text.slice(0, 500)}`);
  return JSON.parse(match[0]);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// --- OpenRouter (free tier — default for POC) ---

export class OpenRouterProvider {
  constructor(apiKey, model, { rpm } = {}) {
    this.apiKey = apiKey;
    this.model = model || "nousresearch/hermes-3-llama-3.1-405b:free";
    this.name = "openrouter";
    this.baseUrl = "https://openrouter.ai/api/v1/chat/completions";
    this.lastRequestAt = 0;
    this.minInterval = rpm ? 60000 / rpm : 6000; // default 10 RPM
    this.forcedWaitUntil = 0;
  }

  async evaluate(systemPrompt, userPrompt) {
    // Proactive pacing
    await this._pace();

    const maxRetries = 5;
    let backoff = 30000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const response = await fetch(this.baseUrl, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
          "HTTP-Referer": "https://github.com/safety-quotient",
          "X-Title": "SafetyQuotient"
        },
        body: JSON.stringify({
          model: this.model,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt }
          ],
          max_tokens: 4096,
          temperature: 0.1
        })
      });

      this.lastRequestAt = Date.now();
      this._checkRateLimitHeaders(response);

      if (response.status === 429) {
        if (attempt === maxRetries) {
          throw new Error(`OpenRouter 429: rate limited after ${maxRetries} retries`);
        }
        const retryAfter = this._getRetryDelay(response, backoff);
        console.log(`  [openrouter] 429 — retry ${attempt}/${maxRetries} in ${(retryAfter / 1000).toFixed(0)}s`);
        await sleep(retryAfter);
        backoff *= 2;
        continue;
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(`OpenRouter error: ${JSON.stringify(data.error)}`);
      }

      const text = data.choices[0].message.content;
      return { raw: text, parsed: parseJSON(text) };
    }
  }

  async _pace() {
    // Respect forced wait from rate-limit headers
    const now = Date.now();
    if (this.forcedWaitUntil > now) {
      const wait = this.forcedWaitUntil - now;
      console.log(`  [openrouter] rate-limit header wait ${(wait / 1000).toFixed(1)}s`);
      await sleep(wait);
    }

    // Standard interval pacing
    const elapsed = Date.now() - this.lastRequestAt;
    const wait = this.minInterval - elapsed;
    if (wait > 0) {
      console.log(`  [openrouter] pacing ${(wait / 1000).toFixed(1)}s`);
      await sleep(wait);
    }
  }

  _checkRateLimitHeaders(response) {
    const remaining = response.headers.get("x-ratelimit-remaining");
    const reset = response.headers.get("x-ratelimit-reset");
    if (remaining === "0" && reset) {
      const resetTime = new Date(reset).getTime();
      if (!isNaN(resetTime)) {
        this.forcedWaitUntil = resetTime;
      }
    }
  }

  _getRetryDelay(response, defaultMs) {
    const retryAfter = response.headers.get("retry-after");
    if (retryAfter) {
      const secs = parseInt(retryAfter, 10);
      if (!isNaN(secs)) return secs * 1000;
    }
    const reset = response.headers.get("x-ratelimit-reset");
    if (reset) {
      const resetTime = new Date(reset).getTime();
      if (!isNaN(resetTime)) {
        const delay = resetTime - Date.now();
        if (delay > 0) return delay;
      }
    }
    return defaultMs;
  }
}

// --- Anthropic Claude ---

export class ClaudeProvider {
  constructor(apiKey, { rpm } = {}) {
    this.client = new Anthropic({ apiKey });
    this.name = "claude";
    this.lastRequestAt = 0;
    this.minInterval = rpm ? 60000 / rpm : 500; // default 120 RPM
  }

  async evaluate(systemPrompt, userPrompt) {
    await this._pace();

    const maxRetries = 5;
    let backoff = 30000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await this.client.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 4096,
          system: systemPrompt,
          messages: [
            { role: "user", content: userPrompt }
          ]
        });

        this.lastRequestAt = Date.now();
        const text = response.content[0].text;
        return { raw: text, parsed: parseJSON(text) };
      } catch (err) {
        this.lastRequestAt = Date.now();
        const isRateLimit = err.status === 429 || err.message?.includes("429");
        if (isRateLimit && attempt < maxRetries) {
          console.log(`  [claude] 429 — retry ${attempt}/${maxRetries} in ${(backoff / 1000).toFixed(0)}s`);
          await sleep(backoff);
          backoff *= 2;
          continue;
        }
        throw err;
      }
    }
  }

  async _pace() {
    const elapsed = Date.now() - this.lastRequestAt;
    const wait = this.minInterval - elapsed;
    if (wait > 0) {
      console.log(`  [claude] pacing ${(wait / 1000).toFixed(1)}s`);
      await sleep(wait);
    }
  }
}

// --- Cloudflare Workers AI ---

export class WorkersAIProvider {
  constructor(accountId, apiToken, { rpm } = {}) {
    this.accountId = accountId;
    this.apiToken = apiToken;
    this.name = "workersai";
    this.model = "@cf/meta/llama-3.1-70b-instruct";
    this.lastRequestAt = 0;
    this.minInterval = rpm ? 60000 / rpm : 1000; // default 60 RPM
  }

  async evaluate(systemPrompt, userPrompt) {
    await this._pace();

    const url = `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.model}`;

    const maxRetries = 5;
    let backoff = 30000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiToken}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt }
          ],
          max_tokens: 4096
        })
      });

      this.lastRequestAt = Date.now();

      if (response.status === 429) {
        if (attempt === maxRetries) {
          throw new Error(`Workers AI 429: rate limited after ${maxRetries} retries`);
        }
        const retryAfter = response.headers.get("retry-after");
        const delay = retryAfter ? parseInt(retryAfter, 10) * 1000 : backoff;
        console.log(`  [workersai] 429 — retry ${attempt}/${maxRetries} in ${(delay / 1000).toFixed(0)}s`);
        await sleep(delay);
        backoff *= 2;
        continue;
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(`Workers AI error: ${JSON.stringify(data.errors)}`);
      }

      const text = data.result.response;
      return { raw: text, parsed: parseJSON(text) };
    }
  }

  async _pace() {
    const elapsed = Date.now() - this.lastRequestAt;
    const wait = this.minInterval - elapsed;
    if (wait > 0) {
      console.log(`  [workersai] pacing ${(wait / 1000).toFixed(1)}s`);
      await sleep(wait);
    }
  }
}

// --- Factory ---

export function createProvider(name, env, { rpm } = {}) {
  const opts = rpm ? { rpm } : {};
  switch (name) {
    case "openrouter":
      if (!env.OPENROUTER_API_KEY) throw new Error("Set OPENROUTER_API_KEY in .env");
      return new OpenRouterProvider(env.OPENROUTER_API_KEY, env.OPENROUTER_MODEL, opts);

    case "claude":
      if (!env.ANTHROPIC_API_KEY) throw new Error("Set ANTHROPIC_API_KEY in .env");
      return new ClaudeProvider(env.ANTHROPIC_API_KEY, opts);

    case "workersai":
      if (!env.CLOUDFLARE_ACCOUNT_ID || !env.CLOUDFLARE_API_TOKEN) {
        throw new Error("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env");
      }
      return new WorkersAIProvider(env.CLOUDFLARE_ACCOUNT_ID, env.CLOUDFLARE_API_TOKEN, opts);

    case "student":
      return new StudentProvider();

    default:
      throw new Error(`Unknown provider: ${name}. Use: openrouter, claude, workersai, student`);
  }
}
