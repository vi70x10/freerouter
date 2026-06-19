import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  canonicalizeModelId,
  stalenessDecay,
  validateComposite,
} from '../../db/benchmark-scores.js';

// ── canonicalizeModelId ─────────────────────────────────────────────────────
describe('canonicalizeModelId', () => {
  it('strips provider prefix and lowercases', () => {
    expect(canonicalizeModelId('meta/Llama-3.3-70B')).toBe('llama-3-3-70b');
    expect(canonicalizeModelId('openai/gpt-5')).toBe('gpt-5');
  });

  it('removes -instruct suffix', () => {
    expect(canonicalizeModelId('meta/llama-3.3-70b-instruct')).toBe('llama-3-3-70b');
  });

  it('removes -chat suffix', () => {
    expect(canonicalizeModelId('google/gemini-3.1-pro-chat')).toBe('gemini-3-1-pro');
  });

  it('removes -base suffix', () => {
    expect(canonicalizeModelId('meta/llama-3.1-8b-base')).toBe('llama-3-1-8b');
  });

  it('removes -preview suffix', () => {
    expect(canonicalizeModelId('qwen/qwen3.6-max-preview')).toBe('qwen3-6-max');
  });

  it('replaces dots with dashes', () => {
    expect(canonicalizeModelId('gpt-5.5')).toBe('gpt-5-5');
    expect(canonicalizeModelId('gemini-3.1-pro')).toBe('gemini-3-1-pro');
  });

  it('handles model IDs without provider prefix', () => {
    expect(canonicalizeModelId('llama-3.3-70b-instruct')).toBe('llama-3-3-70b');
    expect(canonicalizeModelId('gpt-5')).toBe('gpt-5');
  });

  it('removes quantization suffixes like -fp8, -awq, -gptq', () => {
    expect(canonicalizeModelId('llama-3.3-70b-instruct-fp8-fast')).toBe('llama-3-3-70b');
  });

  it('preserves param size like 70b, 8b', () => {
    expect(canonicalizeModelId('llama-3.3-70b')).toBe('llama-3-3-70b');
    expect(canonicalizeModelId('llama-3.1-8b')).toBe('llama-3-1-8b');
  });

  it('the spec example: meta/llama-3.3-70b-instruct → llama-3-3-70b', () => {
    expect(canonicalizeModelId('meta/llama-3.3-70b-instruct')).toBe('llama-3-3-70b');
  });
});

// ── stalenessDecay ──────────────────────────────────────────────────────────
describe('stalenessDecay', () => {
  it('returns 1.0 for a timestamp from right now', () => {
    const now = new Date().toISOString();
    expect(stalenessDecay(now)).toBeCloseTo(1.0, 2);
  });

  it('returns ~0.5 for a timestamp 10 days ago', () => {
    const tenDaysAgo = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString();
    expect(stalenessDecay(tenDaysAgo)).toBeCloseTo(0.5, 2);
  });

  it('returns ~0.25 for a timestamp 20 days ago', () => {
    const twentyDaysAgo = new Date(Date.now() - 20 * 24 * 60 * 60 * 1000).toISOString();
    expect(stalenessDecay(twentyDaysAgo)).toBeCloseTo(0.25, 2);
  });

  it('returns 0 for null/undefined', () => {
    expect(stalenessDecay(null)).toBe(0);
    expect(stalenessDecay(undefined)).toBe(0);
  });

  it('returns 1 for future timestamps', () => {
    const future = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
    expect(stalenessDecay(future)).toBe(1);
  });

  it('uses continuous exponential decay, NOT step functions', () => {
    // 5 days ago should be pow(0.5, 5/10) = pow(0.5, 0.5) ≈ 0.707
    const fiveDaysAgo = new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString();
    expect(stalenessDecay(fiveDaysAgo)).toBeCloseTo(Math.pow(0.5, 0.5), 2);
  });
});

// ── validateComposite ───────────────────────────────────────────────────────
describe('validateComposite', () => {
  it('accepts valid scores in [0, 100]', () => {
    expect(validateComposite(0)).toBe(true);
    expect(validateComposite(50)).toBe(true);
    expect(validateComposite(100)).toBe(true);
    expect(validateComposite(0.01)).toBe(true);
  });

  it('rejects NaN', () => {
    expect(validateComposite(NaN)).toBe(false);
  });

  it('rejects Infinity and -Infinity', () => {
    expect(validateComposite(Infinity)).toBe(false);
    expect(validateComposite(-Infinity)).toBe(false);
  });

  it('rejects scores < 0', () => {
    expect(validateComposite(-0.01)).toBe(false);
    expect(validateComposite(-100)).toBe(false);
  });

  it('rejects scores > 100', () => {
    expect(validateComposite(100.01)).toBe(false);
    expect(validateComposite(200)).toBe(false);
  });
});
