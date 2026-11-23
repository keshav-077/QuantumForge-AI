import recordedRaw from "./recordedScorecard.json";
import type { ScorecardData } from "../types/cryoforge";

export const FALLBACK_SCORECARD: ScorecardData = recordedRaw as unknown as ScorecardData;
