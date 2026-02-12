import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import * as api from "../api";
import { setAuth, clearAuth } from "../auth";

export function useLogin() {
  const navigate = useNavigate();
  return useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      api.login(email, password),
    onSuccess: (data) => {
      setAuth(data.token, data.user);
      navigate({ to: "/" });
    },
  });
}

export function useRegister() {
  const navigate = useNavigate();
  return useMutation({
    mutationFn: ({
      username,
      email,
      password,
    }: {
      username: string;
      email: string;
      password: string;
    }) => api.register(username, email, password),
    onSuccess: (data) => {
      setAuth(data.token, data.user);
      navigate({ to: "/" });
    },
  });
}

export function useLogout() {
  const navigate = useNavigate();
  return () => {
    clearAuth();
    navigate({ to: "/login" });
  };
}
