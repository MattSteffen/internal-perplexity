"use client";

import type { Collection } from "@/lib/types";

interface CollectionSchemaDisplayProps {
  collection: Collection | null;
}

export function CollectionSchemaDisplay({
  collection,
}: CollectionSchemaDisplayProps) {
  if (!collection) {
    return null;
  }

  // Get metadata schema from collection metadata (already parsed in API)
  const metadataSchema = (collection.metadata as { parsed_metadata_schema?: Record<string, unknown> } | undefined)?.parsed_metadata_schema as Record<string, unknown> | undefined;

  // Extract security rules if available
  const securityRules = collection.security_rules || [];

  // Check if schema has properties
  const hasSchemaProperties = metadataSchema && 
    typeof metadataSchema === 'object' && 
    'properties' in metadataSchema && 
    typeof metadataSchema.properties === 'object' &&
    metadataSchema.properties !== null &&
    Object.keys(metadataSchema.properties).length > 0;

  // If no schema and no rules, don't show anything
  if (!hasSchemaProperties && securityRules.length === 0) {
    return null;
  }

  const renderSchemaProperty = (
    key: string,
    property: Record<string, unknown>,
    required: boolean = false,
  ) => {
    const type = property.type as string;
    const description = property.description as string | undefined;
    const enumValues = property.enum as unknown[] | undefined;
    const items = property.items as Record<string, unknown> | undefined;
    const properties = property.properties as Record<string, Record<string, unknown>> | undefined;

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
    <div className="mt-6 space-y-4">
      <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
        Collection Schema & Rules
      </h3>

      {/* Metadata Schema Section */}
      {hasSchemaProperties && metadataSchema && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
            Metadata Schema
          </h4>
          <div className="space-y-2">
            {Object.entries(metadataSchema.properties as Record<string, Record<string, unknown>>).map(
              ([key, property]) => {
                const required = (metadataSchema.required as string[] | undefined)?.includes(key) || false;
                return renderSchemaProperty(key, property, required);
              },
            )}
          </div>
        </div>
      )}

      {/* Security Rules Section */}
      {securityRules.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
            Security Rules
          </h4>
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
      {!hasSchemaProperties && securityRules.length === 0 && (
        <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            No schema or rules defined for this collection.
          </p>
        </div>
      )}
    </div>
  );
}

