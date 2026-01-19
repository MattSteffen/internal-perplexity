"use client";

import type { Collection } from "@/lib/types";
import { Button } from "@/components/ui/button";

interface CollectionInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  collection: Collection | null;
}

export function CollectionInfoModal({
  isOpen,
  onClose,
  collection,
}: CollectionInfoModalProps) {
  if (!isOpen || !collection) return null;

  // Get metadata schema from collection
  const metadataSchema = collection.metadata_schema as Record<string, unknown> | undefined;

  // Extract security rules if available
  const securityRules = collection.security_rules || [];

  // Check if schema has properties
  const hasSchemaProperties =
    metadataSchema &&
    typeof metadataSchema === "object" &&
    "properties" in metadataSchema &&
    typeof metadataSchema.properties === "object" &&
    metadataSchema.properties !== null &&
    Object.keys(metadataSchema.properties).length > 0;

  const renderSchemaProperty = (
    key: string,
    property: Record<string, unknown>,
    required: boolean = false,
  ) => {
    const type = property.type as string;
    const description = property.description as string | undefined;
    const enumValues = property.enum as unknown[] | undefined;
    const items = property.items as Record<string, unknown> | undefined;
    const properties = property.properties as Record<
      string,
      Record<string, unknown>
    > | undefined;

    return (
      <div
        key={key}
        className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-700 dark:bg-zinc-800"
      >
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium text-zinc-900 dark:text-zinc-100">
                {key}
              </span>
              {required && (
                <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/30 dark:text-red-300">
                  Required
                </span>
              )}
              <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                {type}
                {enumValues && ` (${enumValues.length} options)`}
                {items && "[]"}
              </span>
            </div>
            {description && (
              <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                {description}
              </p>
            )}
            {enumValues && enumValues.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {enumValues.map((value, idx) => (
                  <span
                    key={idx}
                    className="rounded bg-zinc-100 px-2 py-0.5 text-xs text-zinc-700 dark:bg-zinc-700 dark:text-zinc-300"
                  >
                    {String(value)}
                  </span>
                ))}
              </div>
            )}
            {items && (
              <div className="mt-2 pl-4 border-l-2 border-zinc-200 dark:border-zinc-700">
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Array items: {items.type as string}
                </p>
              </div>
            )}
            {properties && (
              <div className="mt-2 pl-4 border-l-2 border-zinc-200 dark:border-zinc-700 space-y-2">
                <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                  Nested properties:
                </p>
                {Object.entries(properties).map(([nestedKey, nestedProp]) =>
                  renderSchemaProperty(nestedKey, nestedProp, false),
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 p-6 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            Collection Information: {collection.name}
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
          >
            <svg
              className="h-5 w-5"
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
        </div>

        <div className="space-y-6">
          {/* Statistics Section */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
              Statistics
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
                <div className="text-sm text-zinc-600 dark:text-zinc-400">
                  Documents
                </div>
                <div className="mt-1 text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                  {collection.num_documents ?? 0}
                </div>
              </div>
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
                <div className="text-sm text-zinc-600 dark:text-zinc-400">
                  Chunks
                </div>
                <div className="mt-1 text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                  {collection.num_chunks ?? 0}
                </div>
              </div>
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
                <div className="text-sm text-zinc-600 dark:text-zinc-400">
                  Partitions
                </div>
                <div className="mt-1 text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                  {collection.num_partitions ?? 0}
                </div>
              </div>
            </div>
          </div>

          {/* Description Section */}
          {collection.description && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Description
              </h3>
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
                <p className="text-sm text-zinc-700 dark:text-zinc-300">
                  {collection.description}
                </p>
              </div>
            </div>
          )}

          {/* Access Level Section */}
          {collection.access_level && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Access Level
              </h3>
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
                <span
                  className={`rounded-full px-3 py-1 text-sm font-medium ${
                    collection.access_level === "public"
                      ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                      : collection.access_level === "private"
                      ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300"
                      : "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300"
                  }`}
                >
                  {collection.access_level === "public"
                    ? "Public"
                    : collection.access_level === "private"
                    ? "Private"
                    : "Admin"}
                </span>
              </div>
            </div>
          )}

          {/* Required Roles Section */}
          {collection.required_roles && collection.required_roles.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Required Roles
              </h3>
              <div className="flex flex-wrap gap-2">
                {collection.required_roles.map((role, idx) => (
                  <span
                    key={idx}
                    className="rounded-full bg-purple-100 px-3 py-1 text-sm font-medium text-purple-800 dark:bg-purple-900/30 dark:text-purple-300"
                  >
                    {role}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Metadata Schema Section */}
          {hasSchemaProperties && metadataSchema && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Metadata Schema
              </h3>
              <div className="space-y-2">
                {Object.entries(
                  metadataSchema.properties as Record<
                    string,
                    Record<string, unknown>
                  >,
                ).map(([key, property]) => {
                  const required =
                    (metadataSchema.required as string[] | undefined)?.includes(
                      key,
                    ) || false;
                  return renderSchemaProperty(key, property, required);
                })}
              </div>
            </div>
          )}

          {/* Security Rules Section */}
          {securityRules.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Security Rules
              </h3>
              <div className="space-y-2">
                {securityRules.map((rule, idx) => (
                  <div
                    key={idx}
                    className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-700 dark:bg-zinc-800"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-zinc-900 dark:text-zinc-100">
                            {rule.name}
                          </span>
                          {rule.required && (
                            <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800 dark:bg-red-900/30 dark:text-red-300">
                              Required
                            </span>
                          )}
                        </div>
                        {rule.description && (
                          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                            {rule.description}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!hasSchemaProperties &&
            securityRules.length === 0 &&
            !collection.description &&
            !collection.access_level &&
            (!collection.required_roles ||
              collection.required_roles.length === 0) && (
              <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  No additional information available for this collection.
                </p>
              </div>
            )}
        </div>

        <div className="mt-6 flex justify-end">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
}

