import { graphql } from './client'

describe('graphql client', () => {
  const realFetch = global.fetch

  afterEach(() => {
    global.fetch = realFetch
  })

  function mockFetch(value: unknown) {
    const mock = jest.fn().mockResolvedValue({ json: async () => value })
    global.fetch = mock as unknown as typeof fetch
    return mock
  }

  it('returns data on success', async () => {
    mockFetch({ data: { foo: 'bar' } })
    const result = await graphql<{ foo: string }>('query { foo }')
    expect(result).toEqual({ foo: 'bar' })
  })

  it('throws the first error message when errors are present', async () => {
    mockFetch({ errors: [{ message: 'boom' }, { message: 'second' }] })
    await expect(graphql('query')).rejects.toThrow('boom')
  })

  it('throws when response has no data and no errors', async () => {
    mockFetch({})
    await expect(graphql('query')).rejects.toThrow('Empty response from server')
  })

  it('posts the query and variables as JSON to the GraphQL endpoint', async () => {
    const mock = mockFetch({ data: { ok: true } })
    await graphql('query Q($x:Int){ foo(x:$x) }', { x: 1 })
    expect(mock).toHaveBeenCalledTimes(1)
    const [url, init] = mock.mock.calls[0]
    expect(url).toBe('http://localhost:8080/graphql')
    expect(init.method).toBe('POST')
    expect(init.headers).toEqual({ 'content-type': 'application/json' })
    expect(JSON.parse(init.body as string)).toEqual({
      query: 'query Q($x:Int){ foo(x:$x) }',
      variables: { x: 1 }
    })
  })
})
