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

import React from 'react';

interface IconProps {
  className?: string;
  size?: number;
}

interface IconBaseProps extends IconProps {
  children: React.ReactNode;
}

const IconBase: React.FC<IconBaseProps> = ({ className, size = 20, children }) => (
  <svg 
    className={className} 
    width={size} 
    height={size}
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    {children}
  </svg>
);

export const RagIcon: React.FC<IconProps> = ({ className, size = 20 }) => (
  <svg 
    className={className} 
    width={size} 
    height={size}
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

export const FeaturesIcon: React.FC<IconProps> = ({ className, size }) => (
  <IconBase className={className} size={size}>
    <rect x="3" y="6" width="8" height="4" rx="2" />
    <circle cx="9" cy="8" r="1" fill="currentColor" />
    <rect x="3" y="14" width="8" height="4" rx="2" />
    <circle cx="5" cy="16" r="1" fill="currentColor" />
    <rect x="13" y="10" width="8" height="4" rx="2" />
    <circle cx="19" cy="12" r="1" fill="currentColor" />
  </IconBase>
);

export const ModelsIcon: React.FC<IconProps> = ({ className, size }) => (
  <IconBase className={className} size={size}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
  </IconBase>
);

export const EndpointsIcon: React.FC<IconProps> = ({ className, size }) => (
  <IconBase className={className} size={size}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
  </IconBase>
);

export const AdvancedIcon: React.FC<IconProps> = ({ className, size }) => (
  <IconBase className={className} size={size}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </IconBase>
);

// Backward compatibility: export the old inline function style as well
/* eslint-disable react-refresh/only-export-components */
export const ICON_rag = () => <RagIcon />;
export const ICON_features = () => <FeaturesIcon />;
export const ICON_models = () => <ModelsIcon />;
export const ICON_endpoints = () => <EndpointsIcon />;
export const ICON_advanced = () => <AdvancedIcon />;
