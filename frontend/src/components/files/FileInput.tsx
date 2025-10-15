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

import { useFileUpload } from "../../hooks/useFileUpload";

interface FileInputProps {
  multiple?: boolean;
  accept?: string;
  className?: string;
}

export const FileInput = ({ 
  multiple = true, 
  accept,
  className = "block w-full text-sm file:mr-4 file:rounded file:border-0 file:bg-[var(--nv-green)] file:px-4 file:py-2 file:text-sm"
}: FileInputProps) => {
  const { handleInputChange } = useFileUpload();

  return (
    <input
      type="file"
      multiple={multiple}
      accept={accept}
      onChange={handleInputChange}
      className={className}
    />
  );
}; 