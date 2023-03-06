import { inflate } from 'fflate'

import wasm, { size as wasmSize } from '../target/wasm32-wasi/release/tokenizer.wasm'

import { WebModule } from './webmodule.js'
export { WebModule }

export interface LoadTokenizerTiktoken {
    bpe: string
    special_bpe: [string, number][]
    regex: string
}

export type LoadTokenizerInput = LoadTokenizerTiktoken

export class Tokenizer {
    private webm: WebModule

    private constructor(webm: WebModule) {
        this.webm = webm
    }

    public static async create() {
        const base = Uint8Array.from(atob(wasm), (c) => c.charCodeAt(0))
        const data = await new Promise<Uint8Array>((resolve, reject) => {
            inflate(
                base,
                {
                    consume: true,
                    size: wasmSize,
                },
                (err, data) => {
                    if (err) {
                        reject(err)
                    } else {
                        resolve(data)
                    }
                }
            )
        })
        const webm = await WebModule.load(data)
        return new Tokenizer(webm)
    }

    public static async createFromData(input: Uint8Array) {
        const webm = await WebModule.load(input)
        return new Tokenizer(webm)
    }

    public load(input: LoadTokenizerInput) {
        this.webm.call('load-tokenizer', input)
    }

    public encode(tokenizer: string, input: string): Uint32Array {
        const result = this.webm.call_raw(
            'encode',
            this.webm.pack.encode({
                name: tokenizer,
                input,
            })
        )
        return new Uint32Array(result.buffer, result.byteOffset, result.byteLength / 4)
    }

    public decode(tokenizer: string, input: Uint32Array): string {
        const result = this.webm.call_raw(
            'decode',
            this.webm.pack.encode({
                name: tokenizer,
                input: new Uint8Array(input.buffer),
            })
        )
        return new TextDecoder().decode(result)
    }
}
