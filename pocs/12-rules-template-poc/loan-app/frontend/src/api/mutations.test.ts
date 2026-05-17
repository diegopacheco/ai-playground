import { requestAutoLoan } from './mutations'

describe('requestAutoLoan', () => {
  const realFetch = global.fetch
  afterEach(() => { global.fetch = realFetch })

  it('unwraps requestAutoLoan from the GraphQL response', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      json: async () => ({
        data: {
          requestAutoLoan: {
            approved: true,
            monthlyPayment: 495.03,
            interestRate: 7,
            reason: 'Approved.'
          }
        }
      })
    }) as unknown as typeof fetch

    const decision = await requestAutoLoan({
      amount: 25000,
      termMonths: 60,
      annualIncome: 80000,
      vehicleValue: 30000,
      creditScore: 720
    })

    expect(decision.approved).toBe(true)
    expect(decision.monthlyPayment).toBe(495.03)
    expect(decision.interestRate).toBe(7)
    expect(decision.reason).toBe('Approved.')
  })

  it('sends input as variables under key "input"', async () => {
    const mock = jest.fn().mockResolvedValue({
      json: async () => ({
        data: { requestAutoLoan: { approved: false, monthlyPayment: 0, interestRate: 0, reason: 'x' } }
      })
    })
    global.fetch = mock as unknown as typeof fetch

    await requestAutoLoan({
      amount: 1,
      termMonths: 1,
      annualIncome: 1,
      vehicleValue: 1,
      creditScore: 1
    })

    const body = JSON.parse(mock.mock.calls[0][1].body)
    expect(body.variables).toEqual({
      input: { amount: 1, termMonths: 1, annualIncome: 1, vehicleValue: 1, creditScore: 1 }
    })
    expect(body.query).toMatch(/mutation RequestAutoLoan/)
  })
})
