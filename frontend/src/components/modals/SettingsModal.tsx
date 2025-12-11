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

import { useCallback } from "react";
import { Stack, Text, Divider } from "@kui/react";
import { ModalContainer } from "./ModalContainer";
import { FeatureWarningModal } from "./FeatureWarningModal";
import { RagConfigSection } from "../settings/RagConfigSection";
import { FeatureTogglesSection } from "../settings/FeatureTogglesSection";
import { ModelsSection } from "../settings/ModelsSection";
import { EndpointsSection } from "../settings/EndpointsSection";
import { AdvancedSection } from "../settings/AdvancedSection";
import MetadataSchemaEditor from "../schema/MetadataSchemaEditor";
import { useFeatureWarning } from "../../hooks/useFeatureWarning";

interface SettingsModalProps {
  onClose: () => void;
}

export default function SettingsModal({ onClose }: SettingsModalProps) {
  const { showModal, showWarning, confirmChange, cancelChange } = useFeatureWarning();

  const handleFeatureToggle = useCallback((featureKey: string, newValue: boolean) => {
    showWarning(featureKey, newValue);
  }, [showWarning]);

  return (
    <>
      <ModalContainer isOpen={true} onClose={onClose} title="Settings">
        <Stack gap="density-lg">
          {/* RAG Configuration */}
          <RagConfigSection />

          {/* Feature Toggles */}
          <FeatureTogglesSection 
            onShowWarning={handleFeatureToggle}
          />

          {/* Model Configuration */}
          <ModelsSection />

          {/* Endpoints Configuration */}
          <EndpointsSection />

          {/* Other Settings */}
          <AdvancedSection />

          {/* Metadata Schema Editor */}
          <Divider />
          <div>
            <Text kind="title/sm" style={{ marginBottom: 'var(--spacing-density-sm)' }}>Metadata Schema</Text>
            <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)', marginBottom: 'var(--spacing-density-md)' }}>
              Configure metadata fields for new collections
            </Text>
            <MetadataSchemaEditor />
          </div>
        </Stack>
      </ModalContainer>

      {/* Feature Warning Modal */}
      {showModal && (
        <FeatureWarningModal
          isOpen={showModal}
          onClose={cancelChange}
          onConfirm={confirmChange}
        />
      )}
    </>
  );
}
