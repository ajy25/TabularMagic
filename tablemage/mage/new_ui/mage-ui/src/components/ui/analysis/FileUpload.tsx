import { useState } from 'react';
import { Upload } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { uploadDataset } from '@/lib/api';

interface FileUploadProps {
  onUploadSuccess: (data: { rows: number; columns: string[] }) => void;
}
export const FileUpload = ({ onUploadSuccess }: FileUploadProps) => {
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [testSize, setTestSize] = useState(0.2);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      
      setSelectedFile(file);
      setError(null);
      setIsUploading(true);
      setUploadProgress(0);
  
      const formData = new FormData();
      formData.append('file', file);
      formData.append('test_size', testSize.toString());
  
      try {
        // Simulate upload progress
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => Math.min(prev + 10, 90));
        }, 100);
  
        const response = await fetch('http://127.0.0.1:5005/api/upload', {
          method: 'POST',
          body: formData,
        });
  
        clearInterval(progressInterval);
  
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Upload failed');
        }
  
        const result = await response.json();
        setUploadProgress(100);
        onUploadSuccess({ rows: result.rows, columns: result.columns });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to upload dataset');
        setSelectedFile(null); // Clear selected file on error
      } finally {
        setIsUploading(false);
        setTimeout(() => setUploadProgress(0), 1000);
      }
    };
  
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Upload className="h-12 w-12 text-gray-400" />
                <div>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                    disabled={isUploading}
                  />
                  <label htmlFor="file-upload">
                    <Button 
                      disabled={isUploading}
                      onClick={() => document.getElementById('file-upload')?.click()}
                      className={selectedFile ? "bg-green-500 hover:bg-green-600" : ""}
                    >
                      {isUploading ? 'Uploading...' : 
                        selectedFile ? `Selected ${selectedFile.name}` : 'Choose File'}
                    </Button>
                  </label>
                  <p className="mt-2 text-gray-500">or drag and drop your CSV file</p>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <span className="text-gray-700">Test Dataset Size:</span>
                <Slider
                  min={0.1}
                  max={0.5}
                  step={0.1}
                  value={[testSize]}
                  onValueChange={(value) => setTestSize(value[0])}
                  className="w-32"
                  disabled={isUploading}
                />
                <span className="text-gray-900 font-medium">{testSize}</span>
              </div>
            </div>

            {uploadProgress > 0 && (
              <div className="mt-4">
                <div className="h-2 bg-gray-200 rounded-full">
                  <div
                    className="h-2 bg-blue-500 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  };