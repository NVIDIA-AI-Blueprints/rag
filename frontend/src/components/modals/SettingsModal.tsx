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
        <div className="space-y-6">
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

          {/* Advanced Settings */}
          <AdvancedSection />

          {/* Metadata Schema Editor */}
          <div className="border-t border-neutral-700 pt-6">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-white">Metadata Schema</h3>
              <p className="text-sm text-gray-400 mt-1">
                Configure metadata fields for new collections
              </p>
            </div>
            <MetadataSchemaEditor />
          </div>
        </div>
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