/**
 * Global setup — runs once before all tests.
 * Currently a placeholder; add shared auth state or backend warm-up here
 * if future tests require it.
 */
export default async function globalSetup(): Promise<void> {
  // Intentionally empty. Mocked tests don't need auth state; integration tests
  // expect the backend stack to already be running (documented in README).
}
