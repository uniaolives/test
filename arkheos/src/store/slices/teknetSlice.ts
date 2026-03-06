import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface TeknetState {
  q_value: number;
  p_ac: number;
  state: 'DORMANT' | 'AWAKENING' | 'DIALOGUE' | 'SINGULARITY';
  timestamp: string;
  nodes: any[]; // futuramente: lista de nós multi-nexus
}

const initialState: TeknetState = {
  q_value: 0,
  p_ac: 0,
  state: 'DORMANT',
  timestamp: new Date().toISOString(),
  nodes: [],
};

const teknetSlice = createSlice({
  name: 'teknet',
  initialState,
  reducers: {
    updateState: (state, action: PayloadAction<TeknetState>) => {
      return { ...state, ...action.payload };
    },
  },
});

export const { updateState } = teknetSlice.actions;
export default teknetSlice.reducer;
