/* eslint-env node */

import fs from 'node:fs'

import { nodeResolve } from '@rollup/plugin-node-resolve'
import { deflateSync } from 'fflate'
import { minify, swc } from 'rollup-plugin-swc3'

/** @type import('rollup').RollupOptions */
export default {
    input: 'lib/index.ts',
    output: {
        dir: 'dist',
        format: 'es',
        name: 'tokenizer',
        sourcemap: true,
        esModule: true,
        compact: true,
        minifyInternalExports: true,
        strict: true,
    },
    treeshake: false,
    external: ['worker_threads', 'fflate'],
    plugins: [
        {
            name: 'embed-wasm',
            transform(code, id) {
                if (id.endsWith('.wasm')) {
                    const data = fs.readFileSync(id)
                    const comp = deflateSync(data, {
                        level: 9,
                        mem: 12,
                    })
                    const base = Buffer.from(comp).toString('base64')
                    return {
                        code: `export const size = ${data.length};export default '${base}';`,
                        map: null,
                    }
                }
            },
        },
        nodeResolve({
            rootDir: 'node_modules',
        }),
        swc({
            tsconfig: 'tsconfig.json',
            swcrc: true,
        }),
        minify({
            mangle: {
                keep_classnames: true,
                toplevel: true,
                safari10: false,
                ie8: false,
            },
            compress: {},
            sourceMap: true,
        }),
    ],
}
