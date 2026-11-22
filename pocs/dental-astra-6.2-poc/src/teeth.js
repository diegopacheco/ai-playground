const names = [
  'Central incisor',
  'Lateral incisor',
  'Canine',
  'First premolar',
  'Second premolar',
  'First molar',
  'Second molar',
  'Third molar',
]
const descriptions = {
  Incisor: [
    'Cuts food into smaller pieces at the start of a bite.',
    'A thin, chisel-shaped cutting edge forms the front of the crown.',
  ],
  Canine: [
    'Grips and tears food with a pointed cusp.',
    'A single prominent cusp sits at the corner of the dental arch.',
  ],
  Premolar: [
    'Crushes and breaks down food before grinding.',
    'Raised cusps form a chewing surface between the canine and molars.',
  ],
  Molar: [
    'Grinds and crushes food during chewing.',
    'A broad chewing surface has several cusps separated by grooves.',
  ],
}
export const teeth = Array.from({ length: 32 }, (_, index) => {
  const id = index + 1
  const upper = id <= 16
  const right = id <= 8 || id >= 25
  const position = upper
    ? Math.min(Math.abs(id - 8), Math.abs(id - 9))
    : Math.min(Math.abs(id - 24), Math.abs(id - 25))
  const family =
    position < 2
      ? 'Incisor'
      : position === 2
        ? 'Canine'
        : position < 5
          ? 'Premolar'
          : 'Molar'
  const [role, feature] = descriptions[family]
  return {
    id,
    upper,
    right,
    position,
    family,
    name: names[position],
    arch: upper ? 'Maxillary' : 'Mandibular',
    side: right ? 'Right' : 'Left',
    role,
    feature:
      position === 7
        ? 'The wisdom tooth is the last molar at the back of its quadrant.'
        : feature,
  }
})
