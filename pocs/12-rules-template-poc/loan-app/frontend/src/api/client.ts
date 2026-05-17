const ENDPOINT = 'http://localhost:8080/graphql'

type GraphQLResponse<T> = {
  data?: T
  errors?: Array<{ message: string }>
}

export async function graphql<T>(
  query: string,
  variables?: Record<string, unknown>
): Promise<T> {
  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query, variables })
  })
  const body = (await res.json()) as GraphQLResponse<T>
  if (body.errors && body.errors.length > 0) {
    throw new Error(body.errors[0].message)
  }
  if (!body.data) {
    throw new Error('Empty response from server')
  }
  return body.data
}
