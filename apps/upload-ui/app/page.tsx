"use client";

import { useEffect, useState } from "react";
import { CollectionsList } from "@/components/CollectionsList";
import { FileUpload } from "@/components/FileUpload";
import { MetadataPreview } from "@/components/MetadataPreview";
import { SecurityUserForm } from "@/components/SecurityUserForm";
import { FileMetadataEditor } from "@/components/FileMetadataEditor";
import { SubmitSection } from "@/components/SubmitSection";
import { CollectionSchemaDisplay } from "@/components/CollectionSchemaDisplay";
import { CreateCollectionModal } from "@/components/CreateCollectionModal";
import { CollectionInfoModal } from "@/components/CollectionInfoModal";
import { LoginModal } from "@/components/LoginModal";
import { SearchPanel } from "@/components/SearchPanel";
import { Header } from "@/components/Header";
import { Button } from "@/components/ui/button";
import { fetchCollections, createCollection } from "@/lib/api";
import { useAuth, useSearch, useDocumentUpload } from "@/lib/hooks";
import type { Collection } from "@/lib/types";

// --- Main Component ---

export default function Home() {
  const auth = useAuth();

  // Data State
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [loadingCollections, setLoadingCollections] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [collectionInfoModalOpen, setCollectionInfoModalOpen] = useState(false);
  const [modalSelectedFile, setModalSelectedFile] = useState<File | null>(null);

  // Logic Hooks
  const search = useSearch(selectedCollection);
  const upload = useDocumentUpload(selectedCollection, () => loadCollections());

  const loadCollections = async () => {
    setLoadingCollections(true);
    try {
      const data = await fetchCollections();
      setCollections(Array.isArray(data) ? data : []);
    } catch (e) { console.error(e); }
    finally { setLoadingCollections(false); }
  };

  useEffect(() => {
    if (auth.isAuthenticated) loadCollections();
  }, [auth.isAuthenticated]);

  if (!auth.isAuthenticated) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-black">
        <LoginModal isOpen={auth.isLoginModalOpen} onSubmit={auth.login} />
      </div>
    );
  }

  const isMetadataMode = upload.config.metadata || upload.config.security;
  const showPreview = upload.selectedFile && !upload.isProcessing && isMetadataMode;
  const showProcessing = upload.isProcessing || Array.from(upload.statusMap.values()).some(s => s.isProcessing);

  return (
    <main className="min-h-screen bg-zinc-50 dark:bg-black p-4 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-8xl">
        <Header username={auth.username} />

        {/* Global Error Display */}
        {upload.error && (
          <div className="mb-6 p-4 bg-red-50 text-red-800 rounded-lg border border-red-200">
            {upload.error}
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-3">

          {/* Column 1: Collections */}
          <div>
            <CollectionsList
              collections={collections}
              selectedCollection={selectedCollection}
              onSelectCollection={(n) => setSelectedCollection(n === selectedCollection ? null : n)}
              onRefresh={loadCollections}
              isRefreshing={loadingCollections}
              onCreateCollection={() => setCreateModalOpen(true)}
            />
          </div>

          {/* Column 2: Upload Logic */}
          <div>
            <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-100">Upload Document</h2>

            {showProcessing ? (
              <div className="p-8 bg-white border rounded-lg text-center dark:bg-zinc-900 dark:border-zinc-800">
                <div className="animate-spin h-8 w-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
                <p>{upload.uploadProgress ? `Uploading ${upload.uploadProgress.current}/${upload.uploadProgress.total}` : "Processing..."}</p>
              </div>
            ) : (
              <>
                <FileUpload
                  onFileSelect={upload.handleFileSelect}
                  disabled={!selectedCollection}
                  clearFiles={upload.clearDropzone}
                />

                {selectedCollection && (
                  <div className="mt-4 p-4 bg-white border rounded-lg dark:bg-zinc-900 dark:border-zinc-800 space-y-2">
                    <h3 className="font-semibold text-sm">Options</h3>
                    {(['security', 'metadata', 'upload'] as const).map(key => (
                      <label key={key} className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={upload.config[key]}
                          onChange={e => upload.setConfig(p => ({ ...p, [key]: e.target.checked }))}
                          className="rounded border-zinc-300 text-blue-600"
                        />
                        <span className="text-sm capitalize">{key === 'security' ? 'Add extra security roles' : `Confirm ${key}`}</span>
                      </label>
                    ))}
                  </div>
                )}

                {/* Inline Editing Tools */}
                {upload.selectedFiles.length > 0 && !upload.isProcessing && (
                  <>
                    {!upload.config.security && (
                      <SecurityUserForm
                        selectedGroupId={upload.globalSecurity.groupId}
                        selectedUserIds={upload.globalSecurity.userIds}
                        onGroupChange={id => upload.setGlobalSecurity(p => ({ ...p, groupId: id }))}
                        onUsersChange={ids => upload.setGlobalSecurity(p => ({ ...p, userIds: ids }))}
                        onCreateGroup={() => alert("Not implemented")}
                      />
                    )}

                    {upload.selectedFiles.length > 1 && upload.config.metadata && (
                      <FileMetadataEditor
                        files={upload.selectedFiles.map(f => ({ file: f, metadata: upload.metadataMap.get(f) || {} }))}
                        onMetadataChange={(idx, meta) => upload.updateMetadata(upload.selectedFiles[idx], meta)}
                      />
                    )}

                    <SubmitSection
                      files={upload.selectedFiles}
                      onSubmit={upload.handleUpload}
                      onCancel={upload.resetState}
                      isUploading={upload.isUploading}
                      disabled={false} // Simplification: Let validation inside handleUpload handle errors
                    />
                  </>
                )}
              </>
            )}

            {selectedCollection && (
              <div className="mt-4">
                <Button
                  variant="outline"
                  onClick={() => setCollectionInfoModalOpen(true)}
                  className="w-full"
                >
                  Show Collection Info
                </Button>
              </div>
            )}
          </div>

          {/* Column 3: Search */}
          <SearchPanel selectedCollection={selectedCollection} search={search} />
        </div>

        {/* Modals */}
        {showPreview && (
          <MetadataPreview
            metadata={upload.processedMetadata || {}}
            fileName={upload.selectedFile?.name || ""}
            fileSize={upload.selectedFile?.size || 0}
            onConfirm={(meta) => {
              if (upload.selectedFile) upload.updateMetadata(upload.selectedFile, meta);
              upload.handleUpload();
            }}
            onCancel={() => upload.setSelectedFile(null)}
            isUploading={upload.isUploading}
            showSecurity={upload.config.security}

            // Security Props
            selectedGroupId={modalSelectedFile ? upload.securityMap.get(modalSelectedFile)?.groupId || null : upload.globalSecurity.groupId}
            selectedUserIds={modalSelectedFile ? upload.securityMap.get(modalSelectedFile)?.userIds || [] : upload.globalSecurity.userIds}

            // Handlers
            onGroupChange={(id) => modalSelectedFile
              ? upload.updateSecurity(modalSelectedFile, id, upload.securityMap.get(modalSelectedFile)?.userIds || [])
              : upload.setGlobalSecurity(prev => ({ ...prev, groupId: id }))}

            onUsersChange={(ids) => modalSelectedFile
              ? upload.updateSecurity(modalSelectedFile, upload.securityMap.get(modalSelectedFile)?.groupId || null, ids)
              : upload.setGlobalSecurity(prev => ({ ...prev, userIds: ids }))}

            onCreateGroup={() => alert("Not implemented")}

            // List Handling
            files={upload.selectedFiles.length > 1 ? upload.selectedFiles.map(f => ({
              file: f,
              metadata: upload.metadataMap.get(f) || {},
              security: upload.securityMap.get(f) || { groupId: null, userIds: [] },
              isProcessing: upload.statusMap.get(f)?.isProcessing || false,
              isReady: upload.statusMap.get(f)?.isReady || false
            })) : undefined}

            onMetadataUpdate={upload.updateMetadata}
            onSecurityUpdate={upload.updateSecurity}
            onAllDocumentsMetadataUpdate={(m) => upload.selectedFiles.forEach(f => upload.updateMetadata(f, m))}
            onAllDocumentsSecurityUpdate={(gid, uids) => upload.selectedFiles.forEach(f => upload.updateSecurity(f, gid, uids))}

            selectedFile={modalSelectedFile}
            onSelectedFileChange={setModalSelectedFile}
          />
        )}

        <CreateCollectionModal
          isOpen={createModalOpen}
          onClose={() => setCreateModalOpen(false)}
          onCreate={async (d) => { await createCollection(d); loadCollections(); }}
        />

        <CollectionInfoModal
          isOpen={collectionInfoModalOpen}
          onClose={() => setCollectionInfoModalOpen(false)}
          collection={collections.find(c => c.name === selectedCollection) || null}
        />
      </div>
    </main>
  );
}
