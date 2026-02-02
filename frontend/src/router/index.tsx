import { createBrowserRouter } from 'react-router-dom'
import { AppLayout } from '@/components/layout'
import TranscribePage from '@/pages/TranscribePage'
import RealtimePage from '@/pages/RealtimePage'
import HotwordsPage from '@/pages/HotwordsPage'
import ConfigPage from '@/pages/ConfigPage'
import MonitorPage from '@/pages/MonitorPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <TranscribePage />,
      },
      {
        path: 'realtime',
        element: <RealtimePage />,
      },
      {
        path: 'hotwords',
        element: <HotwordsPage />,
      },
      {
        path: 'config',
        element: <ConfigPage />,
      },
      {
        path: 'monitor',
        element: <MonitorPage />,
      },
    ],
  },
])
