const form = document.querySelector('#form')
const result = document.querySelector('#result')
const species = document.querySelector('#species')
const confidence = document.querySelector('#confidence')
const message = document.querySelector('#message')
const bars = document.querySelector('#bars')

for (const input of form.querySelectorAll('input')) {
  const output = document.querySelector(`#${input.name.replaceAll('_', '-')}-value`)
  input.addEventListener('input', () => { output.value = input.value })
}

form.addEventListener('submit', async event => {
  event.preventDefault()
  result.classList.add('loading')
  const values = Object.fromEntries(new FormData(form))
  try {
    const response = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(values)})
    const prediction = await response.json()
    if (!response.ok) throw new Error(prediction.error)
    species.textContent = prediction.species
    confidence.textContent = `${prediction.confidence}%`
    message.textContent = 'The neural network found the strongest match in its learned botanical boundaries.'
    bars.innerHTML = Object.entries(prediction.probabilities).map(([name, value]) => `<div class="bar-row"><div class="bar-label"><span>${name.replace('Iris ', '')}</span><span>${value}%</span></div><div class="track"><div class="fill" style="width:${value}%"></div></div></div>`).join('')
  } catch (error) {
    species.textContent = 'Unable to classify'
    message.textContent = error.message
    confidence.textContent = '—'
    bars.innerHTML = ''
  } finally {
    result.classList.remove('loading')
  }
})

form.requestSubmit()
