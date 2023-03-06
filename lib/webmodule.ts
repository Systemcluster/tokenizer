import { File, OpenFile, WASI } from '@bjorn3/browser_wasi_shim'
import { Encoder } from 'cbor-x'

export class WebModule {
    readonly wasi: WASI
    readonly wasm: WebAssembly.Module
    readonly inst: WebAssembly.Instance
    readonly pack: Encoder

    private constructor(wasi: WASI, wasm: WebAssembly.Module, inst: WebAssembly.Instance) {
        this.wasi = wasi
        this.wasm = wasm
        this.inst = inst
        this.pack = new Encoder({
            variableMapSize: true,
            useRecords: false,
            structuredClone: false,
            tagUint8Array: false,
            useSelfDescribedHeader: false,
            pack: false,
        })
    }

    public static async load(data: BufferSource) {
        const arg: string[] = []
        const env: string[] = []
        const fds = [new OpenFile(new File([])), new OpenFile(new File([])), new OpenFile(new File([]))]
        const wasi = new WASI(arg, env, fds)
        const wasm = await WebAssembly.compile(data)
        const inst = await WebAssembly.instantiate(wasm, { wasi_snapshot_preview1: wasi.wasiImport })
        wasi.initialize({
            exports: {
                memory: inst.exports.memory as WebAssembly.Memory,
                _initialize: () => {
                    // no initialization needed
                },
            },
        })
        return new WebModule(wasi, wasm, inst)
    }

    public get functions() {
        return Object.keys(this.inst.exports).filter((key) => typeof this.inst.exports[key] === 'function' && !key.startsWith('cabi_'))
    }

    public get memory() {
        return this.inst.exports.memory as WebAssembly.Memory
    }

    public get exports() {
        return this.inst.exports
    }

    public call<T = object>(name: string, input: object | string): T | null {
        const func = this.inst.exports[name] as ((ptr: number, len: number) => number) | undefined
        if (!func) {
            throw new Error(`Function ${name} not found`)
        }
        const [inputPtr, inputLen] = typeof input === 'string' ? this.allocate(input) : this.allocate_raw(this.pack.encode(input))
        const output = func(inputPtr, inputLen)
        const [result, success] = this.read_output(output)
        const postFunc = this.inst.exports[`cabi_post_${name}`] as ((ptr: number) => void) | undefined
        if (postFunc) {
            postFunc(output)
        }
        if (success) {
            if (result.length === 0) {
                return null
            }
            return this.pack.decode(result) as T
        } else {
            throw new Error(new TextDecoder().decode(result))
        }
    }

    public call_raw(name: string, input: Uint8Array): Uint8Array {
        const func = this.inst.exports[name] as ((ptr: number, len: number) => number) | undefined
        if (!func) {
            throw new Error(`Function ${name} not found`)
        }
        const [inputPtr, inputLen] = this.allocate_raw(input)
        const output = func(inputPtr, inputLen)
        const [result, success] = this.read_output(output)
        const postFunc = this.inst.exports[`cabi_post_${name}`] as ((ptr: number) => void) | undefined
        if (postFunc) {
            postFunc(output)
        }
        if (success) {
            return result
        } else {
            throw new Error(new TextDecoder().decode(result))
        }
    }

    private read_output(ptr: number): [Uint8Array, boolean] {
        let resultTag = new DataView(this.memory.buffer).getInt32(ptr, true)
        if (resultTag === 0 || resultTag === 1) {
            ptr += 4
        } else {
            resultTag = 0
        }
        const outputPtr = new DataView(this.memory.buffer).getInt32(ptr, true)
        const outputLen = new DataView(this.memory.buffer).getInt32(ptr + 4, true)
        const result = new Uint8Array(this.memory.buffer.slice(outputPtr, outputPtr + outputLen))
        return [result, resultTag === 0]
    }

    private allocate(data: string) {
        if (data.length === 0) {
            return [0, 0]
        }
        const realloc = this.inst.exports['cabi_realloc'] as (ptr: number, offset: number, num: number, len: number) => number
        const encoder = new TextEncoder()
        let ptr = 0
        let allocated = 0
        let total = 0
        while (data.length > 0) {
            ptr = realloc(ptr, allocated, 1, allocated + data.length)
            allocated += data.length
            const { read, written } = encoder.encodeInto(data, new Uint8Array(this.memory.buffer, ptr + total, allocated - total))
            total += written || 0
            data = data.slice(read)
        }
        if (allocated > total) {
            ptr = realloc(ptr, allocated, 1, total)
        }
        return [ptr, total]
    }

    private allocate_raw(data: Uint8Array) {
        if (data.length === 0) {
            return [0, 0]
        }
        const realloc = this.inst.exports['cabi_realloc'] as (ptr: number, offset: number, num: number, len: number) => number
        const ptr = realloc(0, 0, 1, data.length)
        new Uint8Array(this.memory.buffer, ptr, data.length).set(data)
        return [ptr, data.length]
    }
}