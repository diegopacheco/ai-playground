export function computeDateGroups(albums) {
  if (!albums || albums.length === 0) {
    return [];
  }

  const groupMap = new Map();

  albums.forEach(album => {
    const dateGroup = extractDateGroup(album.created_date);

    if (!groupMap.has(dateGroup)) {
      groupMap.set(dateGroup, []);
    }

    groupMap.get(dateGroup).push(album);
  });

  const groups = Array.from(groupMap.entries()).map(([dateGroup, albums]) => ({
    dateGroup,
    albums: albums.sort((a, b) => a.display_order - b.display_order)
  }));

  return groups.sort((a, b) => {
    if (a.dateGroup === 'unknown') return 1;
    if (b.dateGroup === 'unknown') return -1;
    return b.dateGroup.localeCompare(a.dateGroup);
  });
}

function extractDateGroup(dateString) {
  if (!dateString) {
    return 'unknown';
  }

  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      return 'unknown';
    }

    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');

    return `${year}-${month}`;
  } catch {
    return 'unknown';
  }
}

export function formatDateGroup(dateGroup) {
  if (!dateGroup || dateGroup === 'unknown') {
    return 'Unknown Date';
  }

  const parts = dateGroup.split('-');
  if (parts.length !== 2) {
    return 'Unknown Date';
  }

  const year = parts[0];
  const monthNum = parseInt(parts[1], 10);

  if (isNaN(monthNum) || monthNum < 1 || monthNum > 12) {
    return 'Unknown Date';
  }

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  return `${monthNames[monthNum - 1]} ${year}`;
}

export function sortAlbumsByDate(albums) {
  return [...albums].sort((a, b) => {
    if (!a.created_date && !b.created_date) return 0;
    if (!a.created_date) return 1;
    if (!b.created_date) return -1;

    const dateA = new Date(a.created_date);
    const dateB = new Date(b.created_date);

    return dateB.getTime() - dateA.getTime();
  });
}
