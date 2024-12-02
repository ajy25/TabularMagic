'use client'

import { useState, useEffect } from 'react'
import { FileUpload } from '@/components/ui/analysis/FileUpload'
import { ChatSidebar } from '@/components/ui/chat/ChatSidebar'
import AnalysisHistory from '@/components/ui/analysis/AnalysisHistory'
import { getAnalysisHistory, type AnalysisItem } from '@/lib/api'

export default function Home() {
  const [isDatasetLoaded, setIsDatasetLoaded] = useState(false)
  const [analysisItems, setAnalysisItems] = useState<AnalysisItem[]>([])
  const [error, setError] = useState<string | null>(null)

  const fetchAnalysis = async () => {
    try {
      const analysisHistory = await getAnalysisHistory()
      setAnalysisItems(analysisHistory)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analysis history')
      console.error('Error fetching analysis:', err)
    }
  }

  useEffect(() => {
    if (isDatasetLoaded) {
      fetchAnalysis()
      const interval = setInterval(fetchAnalysis, 5000)
      return () => clearInterval(interval)
    }
  }, [isDatasetLoaded])

  useEffect(() => {
    const handleRefresh = () => {
      fetchAnalysis()
    }
    window.addEventListener('refreshAnalysis', handleRefresh)
    return () => window.removeEventListener('refreshAnalysis', handleRefresh)
  }, [])

  const handleUploadSuccess = async (data: { rows: number; columns: string[] }) => {
    setIsDatasetLoaded(true)
  }

  return (
    <div className="h-screen flex">
      <div className="flex-1 flex flex-col p-6">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold mb-2">Dataset Analyzer</h1>
          <p className="text-gray-600">Upload your dataset and chat with the AI analyzer</p>
        </div>
        
        <div className="mb-6">
          <FileUpload onUploadSuccess={handleUploadSuccess} />
        </div>
        
        {error ? (
          <div className="text-red-500 p-4 border rounded-lg bg-red-50">
            {error}
          </div>
        ) : (
          <div className="flex-1">
            <AnalysisHistory items={analysisItems} />
          </div>
        )}
      </div>
      
      <ChatSidebar onAnalysisUpdated={fetchAnalysis} />
    </div>
  )
}