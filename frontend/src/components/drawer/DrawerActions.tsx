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

import { 
  Button, 
  Flex, 
  Spinner 
} from "@kui/react";
import { Trash2, Plus, X } from "lucide-react";

interface DrawerActionsProps {
  onDelete: () => void;
  onAddSource: () => void;
  onCloseUploader: () => void;
  isDeleting?: boolean;
  showUploader?: boolean;
}

export const DrawerActions = ({ 
  onDelete, 
  onAddSource, 
  onCloseUploader,
  isDeleting = false,
  showUploader = false
}: DrawerActionsProps) => (
  <Flex gap="3" justify="stretch" style={{ width: '100%' }}>
    <Button
      kind="secondary"
      color="danger"
      onClick={onDelete}
      disabled={isDeleting}
    >
      {isDeleting ? (
        <>
          <Spinner size="small" description="" />
          Deleting...
        </>
      ) : (
        <>
          <Trash2 size={16} />
          Delete Collection
        </>
      )}
    </Button>
    
    <Button
      color={showUploader ? "neutral" : "brand"}
      onClick={showUploader ? onCloseUploader : onAddSource}
    >
      {showUploader ? <X size={16} /> : <Plus size={16} />}
      {showUploader ? "Close Uploader" : "Add Source to Collection"}
    </Button>
  </Flex>
);
