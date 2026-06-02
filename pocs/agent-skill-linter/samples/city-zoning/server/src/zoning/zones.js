export const ZONES = {
  R1: {
    name: 'Residential Low',
    maxHeight: 35,
    maxLotCoverage: 0.4,
    minLotArea: 5000,
    maxFar: 0.5,
    frontSetback: 25,
    sideSetback: 8,
    rearSetback: 25,
    permittedUses: ['single_family', 'park']
  },
  R3: {
    name: 'Residential High',
    maxHeight: 75,
    maxLotCoverage: 0.6,
    minLotArea: 3000,
    maxFar: 2.0,
    frontSetback: 15,
    sideSetback: 6,
    rearSetback: 20,
    permittedUses: ['single_family', 'multi_family', 'park']
  },
  C1: {
    name: 'Commercial',
    maxHeight: 90,
    maxLotCoverage: 0.8,
    minLotArea: 2000,
    maxFar: 3.0,
    frontSetback: 10,
    sideSetback: 0,
    rearSetback: 10,
    permittedUses: ['retail', 'office', 'restaurant', 'multi_family']
  },
  I1: {
    name: 'Industrial',
    maxHeight: 60,
    maxLotCoverage: 0.7,
    minLotArea: 10000,
    maxFar: 1.5,
    frontSetback: 20,
    sideSetback: 10,
    rearSetback: 15,
    permittedUses: ['warehouse', 'manufacturing', 'office']
  },
  MU: {
    name: 'Mixed Use',
    maxHeight: 80,
    maxLotCoverage: 0.75,
    minLotArea: 2500,
    maxFar: 2.5,
    frontSetback: 10,
    sideSetback: 5,
    rearSetback: 15,
    permittedUses: ['retail', 'office', 'multi_family', 'restaurant']
  }
};
