"use client";

import { useEffect, useState } from "react";

export interface UseAuthReturn {
  isAuthenticated: boolean;
  username: string;
  isLoginModalOpen: boolean;
  login: (username: string, password: string) => void;
}

/**
 * Handles authentication state and localStorage persistence.
 * Checks for saved credentials on mount and exposes login functionality.
 */
export function useAuth(): UseAuthReturn {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState("");
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);

  useEffect(() => {
    const savedUser = localStorage.getItem("username");
    const savedPass = localStorage.getItem("password");
    if (savedUser && savedPass) {
      setUsername(savedUser);
      setIsAuthenticated(true);
    } else {
      setIsLoginModalOpen(true);
    }
  }, []);

  const login = (u: string, p: string) => {
    setUsername(u);
    setIsAuthenticated(true);
    setIsLoginModalOpen(false);
    localStorage.setItem("username", u);
    localStorage.setItem("password", p);
  };

  return { isAuthenticated, username, isLoginModalOpen, login };
}

