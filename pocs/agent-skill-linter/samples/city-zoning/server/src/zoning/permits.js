export function permitFee(proposal, compliant) {
  if (!compliant) {
    return 0;
  }
  let fee = 250;
  if (proposal.floorArea > 10000) {
    fee += 1500;
  } else if (proposal.floorArea > 5000) {
    fee += 750;
  } else if (proposal.floorArea > 2000) {
    fee += 300;
  }
  if (proposal.use === 'manufacturing' || proposal.use === 'warehouse') {
    fee += 500;
  }
  if (proposal.heightFeet > 60) {
    fee += 400;
  }
  return fee;
}
