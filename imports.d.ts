declare module '*.wasm' {
    const data: string
    export const size: number
    export default data
}
