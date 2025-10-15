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

import { useState } from "react";
import { Button, Stack, Text, Flex, Modal, Checkbox } from "@kui/react";

interface FeatureWarningModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (dontShowAgain: boolean) => void;
}

export const FeatureWarningModal = ({ isOpen, onClose, onConfirm }: FeatureWarningModalProps) => {
  const [dontShowAgain, setDontShowAgain] = useState(false);
  
  const handleConfirm = () => {
    onConfirm(dontShowAgain);
    setDontShowAgain(false);
  };

  if (!isOpen) return null;

  return (
    <Modal
      open={isOpen}
      onOpenChange={onClose}
      slotHeading="Feature Requirement"
      slotFooter={
        <Flex gap="density-md" justify="end" className="w-full">
          <Button kind="secondary" onClick={onClose}>Cancel</Button>
          <Button kind="primary" onClick={handleConfirm}>Enable Anyway</Button>
        </Flex>
      }
    >
      <Stack gap="density-xl">
        <Text kind="body/regular/md">
          Your model needs to have this feature enabled in order for this setting to work properly. 
          Please ensure your model supports this capability before enabling.
        </Text>
        
        <Checkbox
          slotLabel="Don't show this message again"
          checked={dontShowAgain}
          onCheckedChange={(checked) => {
            setDontShowAgain(checked === true);
          }}
        />
      </Stack>
    </Modal>
  );
}; 