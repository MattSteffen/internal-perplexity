# Upload UI Overview

This package contains the Next.js upload UI for managing collections, uploading documents, and searching data against the backend API.

## Top-level files
- `.dockerignore`: Docker build context exclusions.
- `.gitignore`: Git ignore rules for local artifacts.
- `bun.lock`: Bun lockfile for frontend dependencies.
- `components.json`: shadcn/ui configuration.
- `Dockerfile`: Container build for the upload UI.
- `eslint.config.mjs`: ESLint configuration for the project.
- `next.config.ts`: Next.js configuration.
- `package.json`: Project scripts and dependencies.
- `postcss.config.mjs`: PostCSS configuration.
- `README.md`: Project usage notes.
- `TODO.md`: Local project task list.
- `tsconfig.json`: TypeScript configuration.
- `overview.md`: This documentation file.

## `app/`
- `favicon.ico`: App favicon.
- `globals.css`: Global styles.
- `layout.tsx`: Root layout component.
- `page.tsx`: Root page component.

## `components/`
- `CollectionInfoModal.tsx`: Modal for viewing collection details.
- `CollectionSchemaDisplay.tsx`: Displays collection metadata schema.
- `CollectionsList.tsx`: Collection list and selection UI.
- `CreateCollectionModal.tsx`: Modal for creating a collection. Includes a pipeline template dropdown (from `GET /v1/pipelines`); selecting a template fetches full config via `GET /v1/pipelines/{name}` and populates the crawler config JSON textarea—changing the selection overwrites the textarea with the new template’s config. Also collects collection name, access level (public/private/group_only/admin), access groups (when group_only), and optional crawler config JSON. Sends `access_level`, `access_groups`, and `crawler_config` to `POST /v1/collections`; collection name is set in `crawler_config.database.collection`. Backend validation errors (invalid crawler_config) are shown in the modal on submit.
- `FileMetadataEditor.tsx`: Metadata editor for uploads.
- `FileUpload.tsx`: File selection and upload UI.
- `Header.tsx`: Top navigation/header.
- `LoginModal.tsx`: Login/auth modal.
- `MetadataPreview.tsx`: Preview of parsed metadata.
- `SearchDocumentModal.tsx`: Modal opened when a search result is clicked. Shows document metadata (title, authors, optional date/source) and markdown content.
- `SearchPanel.tsx`: Search UI for collections. Displays results with title (from metadata or first line of markdown) and authors; each result is clickable and opens `SearchDocumentModal`.
- `SecurityUserForm.tsx`: Security group/user selection UI.
- `SubmitSection.tsx`: Upload submit actions.

### `components/ui/`
- `button.tsx`: Button component wrapper.
- `input.tsx`: Input component wrapper.

#### `components/ui/shadcn-io/dropzone/`
- `index.tsx`: Dropzone UI component.

## `lib/`
- `api.ts`: API client for backend endpoints. Includes `fetchPipelines()` (GET /v1/pipelines, no auth) and `fetchPipelineConfig(name)` (GET /v1/pipelines/{name}) for the create-collection pipeline template dropdown and config population.
- `types.ts`: Shared TypeScript types for API payloads and UI state. Includes `PipelineInfo` and `PipelinesResponse` for pipeline listing.
- `utils.ts`: Shared utility helpers. Includes `getDocumentDisplayInfo(doc)` for search result display: returns `displayTitle` (metadata.title or first line of markdown or source or "Untitled") and `displayAuthors` (metadata.author normalized to string, including array form).

### `lib/hooks/`
- `index.ts`: Hook exports.
- `useAuth.ts`: Authentication helper hook.
- `useDocumentUpload.ts`: Upload workflow hook.
- `useSearch.ts`: Search workflow hook.

## `public/`
- `file.svg`: UI icon asset.
- `globe.svg`: UI icon asset.
- `next.svg`: Next.js logo asset.
- `vercel.svg`: Vercel logo asset.
- `window.svg`: UI icon asset.
