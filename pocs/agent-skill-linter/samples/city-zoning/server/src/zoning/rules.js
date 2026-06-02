import { ZONES } from './zones.js';

export function evaluateProposal(zoneCode, proposal) {
  const zone = ZONES[zoneCode];
  if (!zone) {
    throw new Error('unknown zone: ' + zoneCode);
  }
  const violations = [];

  if (!zone.permittedUses.includes(proposal.use)) {
    violations.push({ rule: 'permitted_use', message: proposal.use + ' is not permitted in ' + zone.name });
  }
  if (proposal.heightFeet > zone.maxHeight) {
    violations.push({ rule: 'max_height', message: 'height ' + proposal.heightFeet + ' exceeds ' + zone.maxHeight });
  }
  const coverage = proposal.footprintArea / proposal.lotArea;
  if (coverage > zone.maxLotCoverage) {
    violations.push({ rule: 'lot_coverage', message: 'coverage ' + coverage.toFixed(2) + ' exceeds ' + zone.maxLotCoverage });
  }
  if (proposal.lotArea < zone.minLotArea) {
    violations.push({ rule: 'min_lot_area', message: 'lot area ' + proposal.lotArea + ' below ' + zone.minLotArea });
  }
  const far = proposal.floorArea / proposal.lotArea;
  if (far > zone.maxFar) {
    violations.push({ rule: 'floor_area_ratio', message: 'FAR ' + far.toFixed(2) + ' exceeds ' + zone.maxFar });
  }
  if (proposal.frontSetback < zone.frontSetback) {
    violations.push({ rule: 'front_setback', message: 'front setback ' + proposal.frontSetback + ' below ' + zone.frontSetback });
  }
  if (proposal.sideSetback < zone.sideSetback) {
    violations.push({ rule: 'side_setback', message: 'side setback ' + proposal.sideSetback + ' below ' + zone.sideSetback });
  }
  if (proposal.rearSetback < zone.rearSetback) {
    violations.push({ rule: 'rear_setback', message: 'rear setback ' + proposal.rearSetback + ' below ' + zone.rearSetback });
  }

  return {
    zone: zone.name,
    use: proposal.use,
    compliant: violations.length === 0,
    violations
  };
}
