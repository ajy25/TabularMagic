'use client'

import { Card } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { AnalysisItem } from '@/lib/api'

interface AnalysisHistoryProps {
  items: AnalysisItem[]
}

export default function AnalysisHistory({ items }: AnalysisHistoryProps) {
  return (
    <div className="border bg-gray-50 rounded-lg overflow-hidden flex flex-col" style={{ height: '70vh' }}>
      <div className="bg-white p-4 border-b">
        <h2 className="text-lg font-semibold">Analysis Results</h2>
      </div>
      
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4 max-w-full">
          {items.length === 0 ? (
            <div className="text-gray-500 text-center p-4">
              No analysis results yet. Try asking a question about your data!
            </div>
          ) : (
            items.map((item, index) => (
              <Card key={index} className="p-4 bg-white shadow-sm w-full">
                {item.file_type === 'figure' && (
                  <div className="max-w-full overflow-hidden">
                    <img
                      src={`http://localhost:5005/api/analysis/file/${item.file_name}`}
                      alt="Analysis Figure"
                      className="w-full h-auto rounded-lg"
                      onError={(e) => {
                        console.error('Error loading image:', e);
                        console.log('Failed image path:', e.currentTarget.src);
                      }}
                    />
                  </div>
                )}
                
                {item.file_type === 'table' && (
                  <div 
                    className="max-w-full overflow-x-auto"
                    dangerouslySetInnerHTML={{ __html: item.content || '' }}
                  />
                )}
                
                {item.file_type === 'thought' && (
                  <div className="prose max-w-none">
                    {item.content}
                  </div>
                )}
                
                {item.file_type === 'code' && (
                  <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto text-sm max-w-full">
                    <code className="whitespace-pre-wrap break-all">{item.content}</code>
                  </pre>
                )}
              </Card>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}