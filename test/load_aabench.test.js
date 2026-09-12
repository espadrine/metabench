const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

describe('modelNameFromAA', () => {
  test('all keys are in data/aabench.json at .models[*].name', () => {
    const aabenchPath = path.join(__dirname, '../data/aabench.json');
    const aabench = JSON.parse(fs.readFileSync(aabenchPath, 'utf8'));
    const aaNames = new Set(aabench.models.map((d) => d.name));

    // Load modelNameFromAA from bin/load_aabench.js
    const { modelNameFromAA } = require('../bin/load_aabench');

    const missing = Object.keys(modelNameFromAA).filter((k) => !aaNames.has(k));

    assert.deepStrictEqual(
      missing,
      [],
      `modelNameFromAA keys missing in data/aabench.json: ${missing.join(', ')}`
    );
  });
});

describe('findAAManifests', () => {
  test('extracts path and key from escaped homepage HTML', () => {
    const { findAAManifests } = require('../bin/load_aabench');
    const homepage = 'x\\"manifest\\":{\\"path\\":\\"/data/2a1c2b25faec2404.txt\\",\\"key\\":\\"c320ae7d330a6cea83def23ccdf6cbd23df9be177e2b8946de381de94f7fa867\\"}y';
    assert.deepStrictEqual(findAAManifests(homepage), [
      { path: '/data/2a1c2b25faec2404.txt', key: 'c320ae7d330a6cea83def23ccdf6cbd23df9be177e2b8946de381de94f7fa867' },
    ]);
  });

  test('skips manifest blocks without a path and key', () => {
    const { findAAManifests } = require('../bin/load_aabench');
    const homepage = '\\"manifest\\":\\"$1f:props:children:props:manifest\\"';
    assert.deepStrictEqual(findAAManifests(homepage), []);
  });
});

describe('decryptAAData', () => {
  test('round-trips a gzip-compressed AES-GCM payload', () => {
    const crypto = require('crypto');
    const zlib = require('zlib');
    const { decryptAAData } = require('../bin/load_aabench');

    const key = crypto.randomBytes(32);
    const iv = crypto.createHash('sha256').update(key).digest().slice(0, 12);
    const expected = { models: [{ name: 'Test Model', gpqa: 0.9 }] };
    const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
    const compressed = zlib.gzipSync(Buffer.from(JSON.stringify(expected), 'utf8'));
    const encrypted = Buffer.concat([cipher.update(compressed), cipher.final(), cipher.getAuthTag()]);

    assert.deepStrictEqual(decryptAAData(encrypted, key.toString('hex')), expected);
  });

  test('throws on the wrong key', () => {
    const crypto = require('crypto');
    const zlib = require('zlib');
    const { decryptAAData } = require('../bin/load_aabench');

    const key = crypto.randomBytes(32);
    const iv = crypto.createHash('sha256').update(key).digest().slice(0, 12);
    const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
    const compressed = zlib.gzipSync(Buffer.from('{"models":[]}', 'utf8'));
    const encrypted = Buffer.concat([cipher.update(compressed), cipher.final(), cipher.getAuthTag()]);

    assert.throws(() => decryptAAData(encrypted, crypto.randomBytes(32).toString('hex')));
  });
});
