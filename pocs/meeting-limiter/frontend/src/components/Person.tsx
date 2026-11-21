import { initials, roleLabel, type Role, type User } from "../features/people/types";

const ROLE_COLORS: Readonly<Record<Role, string>> = {
  engineer: "#0f6cbd",
  manager: "#107c41",
  director: "#8a3ffc",
  vp: "#c75b12",
};

export function roleColor(role: Role): string {
  return ROLE_COLORS[role];
}

export function Avatar({ user, size = 28 }: { readonly user: User; readonly size?: number }) {
  return (
    <span className="avatar" style={{ width: size, height: size, background: roleColor(user.role), fontSize: size * 0.4 }} aria-hidden="true">
      {initials(user.name)}
    </span>
  );
}

export function RoleBadge({ role }: { readonly role: Role }) {
  return (
    <span className="role-badge" style={{ color: roleColor(role), borderColor: roleColor(role) }}>
      {roleLabel(role)}
    </span>
  );
}

export function PersonCell({ user }: { readonly user: User }) {
  return (
    <span className="person">
      <Avatar user={user} />
      <span>
        <strong>{user.name}</strong>
        <small>{user.title}</small>
      </span>
    </span>
  );
}
