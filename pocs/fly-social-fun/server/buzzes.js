export const TEMPLATES = {
  'spot:banana': [
    'This banana is ripening at the perfect speed. Brown spots = premium. #BananaLife',
    'Not me spending 6 hours on one banana spot. Worth it. #BananaLife',
    'Ranking fruits: 1. old banana 2. older banana 3. everything else #FoodReview',
    'Just landed on the banana and rubbed my hands together like a villain #Snacking',
    'Banana check-in. The vibes? Fermenting. #BananaLife',
  ],
  'spot:pizza': [
    'Human left pizza unattended. Big mistake. Huge. #PizzaHeist',
    'Pepperoni tastes 30% better when you are not supposed to be here #PizzaHeist',
    'Walked all over the pizza. That is how I taste things. You are welcome. #FlyFacts',
    'Pizza crust is my cardio track #FitFly',
  ],
  'spot:trash': [
    'Trash can just dropped a new flavor. 10/10 would crawl in again #TrashTuesday',
    'People call it garbage. I call it a buffet with a lid. #TrashTuesday',
    'Networking event at the trash can tonight. Bring your own germs. #TrashTuesday',
    'Found a 5-star dumpster experience right here in the kitchen #FoodReview',
  ],
  'spot:mug': [
    'Cold coffee at the bottom of the mug. Caffeine in a 12mg body is WILD #CoffeeBuzz',
    'Did a lap around the mug rim. Almost fell in. Did not. Legend. #CoffeeBuzz',
    'Espresso yourself #CoffeeBuzz',
  ],
  'spot:lamp': [
    'The light. I must go toward the light. #LampLife',
    'Bonked the lamp 47 times. It is a relationship now. #LampLife',
    'Moths think they invented this. Posers. #LampLife',
    'Sunbathing under the lamp. Wings looking crispy. #LampLife',
  ],
  'spot:window': [
    'Been flying into this invisible wall for 3 hours. Wall 1, me 0. #WindowWars',
    'I can SEE outside. Why can I not GO outside. #WindowWars',
    'Day 4 of trying the window. Hope is a dangerous thing. #WindowWars',
    'The outside flies look so free. Anyway, back to bonking. #WindowWars',
  ],
  'spot:web': [
    'Checking out this silky hammock in the corner. Seems legit. #Adventure',
    'Spider said it is a free spa. Going in! #Adventure',
  ],
  generic: [
    'Rubbing my hands together for no reason. Plotting? Maybe. #FlyFacts',
    'I have 4000 lenses per eye and I still cannot find my keys',
    'Life is short. Literally 28 days. Eat the banana. #YOLO',
    'Buzz buzz. That is it. That is the post.',
    'Hot take: humans are just big slow landing pads #FlyFacts',
    'Just threw up on my food so I can eat it. Totally normal. #FlyFacts',
    'My whole generation is 2 weeks old and we already have nostalgia #GenZzz',
    'I see the world in slow motion. Your swatter is a joke to me. #FlyFacts',
  ],
  reply: [
    '@{handle} so true bestie',
    '@{handle} ratio',
    '@{handle} this is the content I hatched for',
    '@{handle} source?',
    '@{handle} brb flying there',
    '@{handle} legend. all 6 legs of you.',
    '@{handle} counterpoint: no',
    '@{handle} I was there first',
    '@{handle} unfollowing. jk. unless?',
  ],
  flirt: [
    '@{handle} are you a rotten peach? Because I would land on you for hours #FlyLove',
    '@{handle} your compound eyes... all 8000 of them... #FlyLove',
    '@{handle} wanna go bonk the window together sometime? #FlyLove',
    '@{handle} you make my wings buzz at 200Hz #FlyLove',
  ],
  couple: [
    '@{handle} it is official. Expecting about 500 eggs. #FlyLove',
    '@{handle} yes. a thousand times yes. meet me at the trash can #FlyLove',
  ],
  hatch: [
    'Just hatched! First buzz ever. Where is the banana? #NewHere',
    'Hi everyone, I was a maggot like 5 minutes ago #NewHere',
    'Born in the trash can, raised by the timeline #NewHere',
  ],
  hatchWithParents: [
    'Just hatched! Shoutout to @{parent} for the genes and the garbage #NewHere',
    'Proud child of @{parent}. I have 499 siblings and they are all loud #NewHere',
  ],
  'rip:swatter': [
    'RIP @{handle}. Swatted at {age} days old. Gone too soon. #SwatterSurvivor',
    'Pour one out for @{handle}. The swatter took another legend. #SwatterSurvivor',
  ],
  'rip:spider': [
    'Last seen heading into the spa corner. RIP @{handle} #SpiderSzn',
    '@{handle} trusted the silky hammock. We warned them. #SpiderSzn',
  ],
  'rip:age': [
    '@{handle} passed at the ripe old age of {age} days. A true elder. #RIP',
    'RIP @{handle}. {age} days of bonking windows. What a run. #RIP',
  ],
  lastWords: [
    'Ok this hammock is VERY sticky. Tell my kids I love them. All 500 of them.',
    'The spa guy has 8 legs. Why does the spa guy have 8 legs.',
  ],
  escapeWeb: [
    'Escaped the spider web with only minor emotional damage #SpiderSzn',
    'Spider blinked first. I am out of here. #SpiderSzn',
  ],
  swatAlert: [
    'SWATTER AT THE {spot}!!! EVERYBODY SCATTER #SwatterAlert',
    'Did you all see that giant hand thing at the {spot}?? #SwatterAlert',
  ],
  swatDodge: [
    'Dodged the swatter at the {spot}. I see in slow-mo, baby. #Matrix',
    'Human swung at me and MISSED. Still shaking. #SwatterSurvivor',
  ],
  swatMiss: [
    'Human just swatted the empty {spot}. Aim assist when? #SwatterFail',
    'Imagine swinging at a {spot} with nobody on it #SwatterFail',
  ],
  snack: [
    'HUMAN DROPPED {snack} ON THE {spot}. THIS IS NOT A DRILL #FreeFood',
    'Breaking: {snack} spotted near the {spot}. Wings up! #FreeFood',
  ],
  milestone: [
    'Just hit {count} followers! Thank you to every single one of my eyes #Blessed',
    '{count} followers?? Stop, I only live for 4 weeks #Blessed',
  ],
};

export function fill(template, values) {
  return template.replace(/\{(\w+)\}/g, (match, key) => {
    if (!(key in values)) throw new Error(`missing value ${key} for template "${template}"`);
    return String(values[key]);
  });
}

export function writeBuzz(kind, rng, values = {}) {
  const templates = TEMPLATES[kind];
  if (!templates) throw new Error(`unknown buzz kind ${kind}`);
  return fill(rng.pick(templates), values);
}

export function spotBuzz(spotId, rng) {
  const kind = `spot:${spotId}`;
  if (TEMPLATES[kind] && rng.chance(0.75)) return writeBuzz(kind, rng);
  return writeBuzz('generic', rng);
}
