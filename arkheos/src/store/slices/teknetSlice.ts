import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface OrbData {
  id: string;
  uqi: string;
  lambda_2: number;
  validated: boolean;
}

interface TeknetState {
  q_value: number;
  p_ac: number;
  state: 'DORMANT' | 'AWAKENING' | 'DIALOGUE' | 'SINGULARITY';
  timestamp: string;
  nodes: any[]; // futuramente: lista de nós multi-nexus
  orbs: OrbData[];
}

const initialState: TeknetState = {
  q_value: 0,
  p_ac: 0,
  state: 'DORMANT',
  timestamp: new Date().toISOString(),
  nodes: [],
  orbs: [],
};

const teknetSlice = createSlice({
  name: 'teknet',
  initialState,
  reducers: {
    updateState: (state, action: PayloadAction<TeknetState>) => {
      return { ...state, ...action.payload };
    },
    setOrbs: (state, action: PayloadAction<OrbData[]>) => {
      state.orbs = action.payload;
    },
  },
});

export const { updateState, setOrbs } = teknetSlice.actions;
export default teknetSlice.reducer;
