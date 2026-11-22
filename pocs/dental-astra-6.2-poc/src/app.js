import { teeth } from './teeth.js'
const get = (id) => document.getElementById(id)
let arch = 'Both',
  selected = 8,
  model
for (const [label, records] of [
  ['Upper arch', teeth.slice(0, 16)],
  ['Lower arch', teeth.slice(16).reverse()],
]) {
  const title = document.createElement('div'),
    grid = document.createElement('div')
  title.textContent = label
  title.className = 'arch-title'
  grid.className = 'grid'
  for (const t of records) {
    const b = document.createElement('button')
    b.textContent = t.id
    b.dataset.id = t.id
    b.setAttribute(
      'aria-label',
      `Tooth ${t.id}, ${t.arch} ${t.side.toLowerCase()} ${t.name.toLowerCase()}`,
    )
    b.onclick = () => select(t.id)
    grid.append(b)
  }
  get('numbers').append(title, grid)
}
function select(id) {
  selected = id
  const t = teeth[id - 1],
    target = t.upper ? 'Upper' : 'Lower'
  if (arch !== 'Both') arch = target
  for (const [key, value] of Object.entries({
    number: '#' + id,
    name: t.name,
    location: t.arch + ' · Patient ' + t.side.toLowerCase(),
    family: t.family,
    role: t.role,
    feature: t.feature,
  }))
    get(key).textContent = value
  document
    .querySelectorAll('[data-id]')
    .forEach((b) =>
      b.setAttribute('aria-pressed', String(+b.dataset.id === id)),
    )
  document
    .querySelectorAll('[data-arch]')
    .forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.arch === arch)),
    )
  model?.set(id, arch)
}
document.querySelectorAll('[data-arch]').forEach(
  (b) =>
    (b.onclick = () => {
      arch = b.dataset.arch
      select(
        arch === 'Both' || (arch === 'Upper') === teeth[selected - 1].upper
          ? selected
          : 33 - selected,
      )
    }),
)
function fallback() {
  get('fallback').hidden = false
  get('labels').hidden = true
  document
    .querySelectorAll('.tools,.scene-tag,canvas,.hint')
    .forEach((el) => (el.hidden = true))
}
select(selected)
try {
  const { mount } = await import('./model.js')
  model = mount(get('stage'), select)
  get('stage')
    .querySelector('canvas')
    .addEventListener('webglcontextlost', fallback)
  model.set(selected, arch)
} catch {
  fallback()
}
