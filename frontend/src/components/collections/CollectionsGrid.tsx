import { VerticalNav, Spinner, StatusMessage, Flex } from "@kui/react";
import { useCollections } from "../../api/useCollectionsApi";
import type { Collection } from "../../types/collections";
import { CollectionItem } from "./CollectionItem";
import { useCollectionsStore } from "../../store/useCollectionsStore";

const CollectionsEmptyIcon = () => (
  <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
  </svg>
);

interface CollectionsGridProps {
  searchQuery: string;
}

const Wrapper = ({ children }: { children: React.ReactNode }) => {
  return (
    <Flex 
      justify="center" 
      align="center" 
      style={{ 
        width: '100%', 
        height: '100%',
        backgroundColor: 'var(--background-color-surface-navigation)',
        borderRight: '1px solid var(--border-color-base)'
      }}
    >
      {children}
    </Flex>
  );
};

export const CollectionsGrid = ({ searchQuery }: CollectionsGridProps) => {
  const { data, isLoading, error } = useCollections();
  const { selectedCollections, toggleCollection } = useCollectionsStore();

  // Frontend filtering and sorting of collections for consistent order
  const filteredCollections = (data || [])
    .filter((collection: Collection) => {
      // Hide system collections from the list
      if (collection.collection_name === "metadata_schema" || collection.collection_name === "meta") {
        return false;
      }
      // Apply search filter
      return collection.collection_name.toLowerCase().includes(searchQuery.toLowerCase());
    })
    .sort((a: Collection, b: Collection) => 
      a.collection_name.toLowerCase().localeCompare(b.collection_name.toLowerCase())
    );

  if (isLoading) {
    return (
      <Wrapper>
        <Spinner description="Loading collections..." />
      </Wrapper>
    );
  }

  if (error) {
    return (
      <Wrapper>
        <StatusMessage
          slotHeading="Failed to load collections"
          slotMedia={<CollectionsEmptyIcon />}
        />
      </Wrapper>
    );
  }

  // Check for search-specific empty state first
  if (!filteredCollections.length && searchQuery) {
    return (
      <Wrapper>
        <StatusMessage
          slotHeading="No matches found"
          slotSubheading={`No collections match "${searchQuery}"`}
          slotMedia={<CollectionsEmptyIcon />}
        />
      </Wrapper>
    );
  }

  // Show general empty state if no collections exist or only system collections exist
  if (!data?.length || !filteredCollections.length) {
    return (
      <Wrapper>
        <StatusMessage
          slotHeading="No collections"
          slotSubheading="Create your first collection and add files to customize your model response."
          slotMedia={<CollectionsEmptyIcon />}
        />
      </Wrapper>
    );
  }

  return (
    <VerticalNav
      style={{ width: '100%' }}
      items={filteredCollections.map((collection: Collection) => ({
        id: collection.collection_name,
        slotLabel: (
          <CollectionItem 
            collection={collection} 
          />
        ),
        active: selectedCollections.includes(collection.collection_name),
        href: `#${collection.collection_name}`,
        attributes: {
          VerticalNavLink: {
            onClick: (e: React.MouseEvent) => {
              e.preventDefault();
              toggleCollection(collection.collection_name);
            }
          }
        }
      }))}
    />
  );
}; 