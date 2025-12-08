"use client";

import { useEffect, useState } from "react";
import { fetchRoles, fetchUsers } from "@/lib/api";
import type { Role, User } from "@/lib/types";

interface SecurityUserFormProps {
    selectedGroupId: string | null;
    selectedUserIds: string[];
    onGroupChange: (groupId: string | null) => void;
    onUsersChange: (userIds: string[]) => void;
    onCreateGroup?: () => void;
}

export function SecurityUserForm({
    selectedGroupId,
    selectedUserIds,
    onGroupChange,
    onUsersChange,
    onCreateGroup,
}: SecurityUserFormProps) {
    const [roles, setRoles] = useState<Role[]>([]);
    const [users, setUsers] = useState<User[]>([]);
    const [isLoadingGroups, setIsLoadingGroups] = useState(true);
    const [isLoadingUsers, setIsLoadingUsers] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showCreateGroup, setShowCreateGroup] = useState(false);
    const [newGroupName, setNewGroupName] = useState("");

    useEffect(() => {
        loadRoles();
        loadUsers();
    }, []);

    const loadRoles = async () => {
        try {
            setIsLoadingGroups(true);
            setError(null);
            const data = await fetchRoles();
            setRoles(data);
        } catch (err) {
            setError(
                err instanceof Error ? err.message : "Failed to load security roles",
            );
        } finally {
            setIsLoadingGroups(false);
        }
    };

    const loadUsers = async () => {
        try {
            setIsLoadingUsers(true);
            setError(null);
            const data = await fetchUsers();
            setUsers(data);
        } catch (err) {
            setError(
                err instanceof Error ? err.message : "Failed to load users",
            );
        } finally {
            setIsLoadingUsers(false);
        }
    };

    const handleGroupSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const value = e.target.value;
        if (value === "__create_new__") {
            setShowCreateGroup(true);
            onGroupChange(null);
        } else {
            setShowCreateGroup(false);
            onGroupChange(value || null);
        }
    };

    const handleUserToggle = (userId: string) => {
        if (selectedUserIds.includes(userId)) {
            // Only allow removal if at least one user remains
            if (selectedUserIds.length > 1) {
                onUsersChange(selectedUserIds.filter((id) => id !== userId));
            }
        } else {
            onUsersChange([...selectedUserIds, userId]);
        }
    };

    const handleCreateGroup = () => {
        if (newGroupName.trim()) {
            onCreateGroup?.();
            setNewGroupName("");
            setShowCreateGroup(false);
        }
    };

    const selectedUsers = users.filter((user) =>
        selectedUserIds.includes(user.id),
    );

    return (
        <div className="mt-6 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
            <h3 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Security & User Assignment
            </h3>

            {error && (
                <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 dark:border-red-800 dark:bg-red-950/30">
                    <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
                </div>
            )}

            <div className="grid gap-6 md:grid-cols-2">
                {/* Left Section: Security Roles */}
                <div>
                    <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                        Security Role
                    </label>
                    {isLoadingGroups ? (
                        <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-500 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400">
                            Loading roles...
                        </div>
                    ) : (
                        <>
                            <select
                                value={selectedGroupId || ""}
                                onChange={handleGroupSelect}
                                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                            >
                                <option value="">Select a security role</option>
                                {roles.map((role, index) => (
                                    <option key={`role-${index}-${role.role}`} value={role.role}>
                                        {role.role} {role.privileges?.length > 0 && `(${role.privileges?.length} privileges)`}
                                    </option>
                                ))}
                                <option value="__create_new__">+ Create new role</option>
                            </select>

                            {showCreateGroup && (
                                <div className="mt-3 flex gap-2">
                                    <input
                                        type="text"
                                        value={newGroupName}
                                        onChange={(e) => setNewGroupName(e.target.value)}
                                        placeholder="Enter group name"
                                        className="flex-1 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                                    />
                                    <button
                                        onClick={handleCreateGroup}
                                        className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600"
                                    >
                                        Create
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* Right Section: Users */}
                <div>
                    <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                        Users (select at least 1)
                    </label>
                    {isLoadingUsers ? (
                        <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-500 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400">
                            Loading users...
                        </div>
                    ) : (
                        <>
                            <div className="max-h-48 overflow-y-auto rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-800">
                                {users.length === 0 ? (
                                    <div className="p-3 text-sm text-zinc-500 dark:text-zinc-400">
                                        No users available
                                    </div>
                                ) : (
                                    <div className="divide-y divide-zinc-200 dark:divide-zinc-700">
                                        {users.map((user) => (
                                            <label
                                                key={user.id}
                                                className="flex cursor-pointer items-center gap-3 p-3 hover:bg-zinc-50 dark:hover:bg-zinc-700/50"
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={selectedUserIds.includes(user.id)}
                                                    onChange={() => handleUserToggle(user.id)}
                                                    disabled={
                                                        selectedUserIds.includes(user.id) &&
                                                        selectedUserIds.length === 1
                                                    }
                                                    className="h-4 w-4 rounded border-zinc-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed dark:border-zinc-600 dark:bg-zinc-800"
                                                />
                                                <span className="flex-1 text-sm text-zinc-900 dark:text-zinc-100">
                                                    {user.name}
                                                    {user.email && (
                                                        <span className="ml-2 text-xs text-zinc-500 dark:text-zinc-400">
                                                            ({user.email})
                                                        </span>
                                                    )}
                                                </span>
                                            </label>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Selected Users Display */}
                            {selectedUsers.length > 0 && (
                                <div className="mt-3">
                                    <p className="mb-2 text-xs font-medium text-zinc-600 dark:text-zinc-400">
                                        Selected users:
                                    </p>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedUsers.map((user) => (
                                            <span
                                                key={user.id}
                                                className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-1 text-xs text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                                            >
                                                {user.name}
                                                {selectedUserIds.length > 1 && (
                                                    <button
                                                        onClick={() => handleUserToggle(user.id)}
                                                        className="ml-1 rounded-full hover:bg-blue-200 dark:hover:bg-blue-800/50"
                                                        aria-label={`Remove ${user.name}`}
                                                    >
                                                        <svg
                                                            className="h-3 w-3"
                                                            fill="none"
                                                            stroke="currentColor"
                                                            viewBox="0 0 24 24"
                                                        >
                                                            <path
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                                strokeWidth={2}
                                                                d="M6 18L18 6M6 6l12 12"
                                                            />
                                                        </svg>
                                                    </button>
                                                )}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

