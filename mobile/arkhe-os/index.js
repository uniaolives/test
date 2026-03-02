import { registerRootComponent } from 'expo';
import App from './App';
import { registerBackgroundSync } from './services/BackgroundSync';

// Register background tasks
registerBackgroundSync();

// Register root component
registerRootComponent(App);
