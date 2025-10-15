// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// src/components/MetadataSchemaEditor.tsx

import { useSchemaEditor } from "../../hooks/useSchemaEditor";
import { FieldsList } from "./FieldsList";
import { NewFieldForm } from "./NewFieldForm";
import { Panel, Text } from "@kui/react";

// Export all schema editor components for external use
export { FieldEditForm } from "./FieldEditForm";
export { FieldDisplayCard } from "./FieldDisplayCard";
export { FieldsList } from "./FieldsList";
export { NewFieldForm } from "./NewFieldForm";

const SchemaIcon = () => (
  <svg 
    className="w-5 h-5 text-[var(--nv-green)]" 
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
    data-testid="schema-icon"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
  </svg>
);

const SchemaContent = () => {
  

  return (
    <>
      <Text kind="body/bold/md">Define metadata fields for this collection.</Text>
  
      <FieldsList />
      {<NewFieldForm />}
    </>
  );
}

export default function MetadataSchemaEditor() {
  const { showSchemaEditor } = useSchemaEditor();
  
  return (
    <Panel
      slotHeading="Metadata Schema"
      slotIcon={<SchemaIcon />}
    >
      {showSchemaEditor && <SchemaContent />}
    </Panel>
  );
}
