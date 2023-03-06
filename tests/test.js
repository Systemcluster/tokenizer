/* eslint-env node */
/* eslint-disable import/no-unresolved */

import test from 'node:test'

import assert from 'node:assert'
import fs from 'node:fs'
import path from 'node:path'
import url from 'node:url'

const __dirname = path.dirname(url.fileURLToPath(import.meta.url))

test('encode', async () => {
    const data = fs.readFileSync(path.resolve(__dirname, './cl100k_base.tiktoken'), 'utf-8')
    const { Tokenizer } = await import('../dist/index.js')

    const tokenizer = await Tokenizer.create()
    tokenizer.load({
        name: 'cl100k',
        variant: {
            LoadTokenizerTiktoken: {
                bpe: data,
                special_bpe: [
                    ['<|fim_prefix|>', 100258],
                    ['<|fim_suffix|>', 100260],
                    ['<|endofprompt|>', 100276],
                    ['<|endoftext|>', 100257],
                    ['<|fim_middle|>', 100259],
                    ['<|im_start|>', 100264],
                    ['<|im_end|>', 100265],
                ],
                // eslint-disable-next-line max-len
                regex: String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
            },
        },
    })

    const tokens1 = tokenizer.encode('cl100k', 'Hello World!')
    console.log('Tokens:', [...tokens1])
    assert.equal(JSON.stringify([...tokens1]), JSON.stringify([9906, 4435, 0]), 'Tokens should be [9906, 4435, 0]')

    const tokens2 = tokenizer.encode('cl100k', 'hello <|endoftext|>')
    console.log('Tokens:', [...tokens2])
    assert.equal(JSON.stringify([...tokens2]), JSON.stringify([15339, 220, 100257]), 'Tokens should be [15339, 220, 100257]')
})

test('decode', async () => {
    const data = fs.readFileSync(path.resolve(__dirname, './cl100k_base.tiktoken'), 'utf-8')
    const { Tokenizer } = await import('../dist/index.js')

    const tokenizer = await Tokenizer.create()
    tokenizer.load({
        name: 'cl100k',
        variant: {
            LoadTokenizerTiktoken: {
                bpe: data,
                special_bpe: [
                    ['<|fim_prefix|>', 100258],
                    ['<|fim_suffix|>', 100260],
                    ['<|endofprompt|>', 100276],
                    ['<|endoftext|>', 100257],
                    ['<|fim_middle|>', 100259],
                    ['<|im_start|>', 100264],
                    ['<|im_end|>', 100265],
                ],
                // eslint-disable-next-line max-len
                regex: String.raw`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
            },
        },
    })

    const string1 = tokenizer.decode('cl100k', new Uint32Array([9906, 4435, 0]))
    console.log('String:', string1)
    assert.equal(string1, 'Hello World!', 'String should be "Hello World!"')

    const string2 = tokenizer.decode('cl100k', new Uint32Array([15339, 220, 100257]))
    console.log('String:', string2)
    assert.equal(string2, 'hello <|endoftext|>', 'String should be "hello <|endoftext|>"')
})
