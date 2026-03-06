import { configureStore } from '@reduxjs/toolkit';
import teknetReducer from './slices/teknetSlice';

export const store = configureStore({
  reducer: {
    teknet: teknetReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
