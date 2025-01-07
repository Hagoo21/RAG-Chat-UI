import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import styles from './knowledge.module.less';
import { classnames } from '../../components/utils';
import { Icon, Button, Title } from '../../components';
import { uploadFiles } from '../service/api';

export function FileList({ files }) {
  return (
    <div className={styles.fileList}>
      {files.map((file, index) => (
        <div key={index} className={styles.fileItem}>
          <div className={styles.filePreview}>
            {file.type.startsWith('image/') ? (
              <img src={URL.createObjectURL(file)} alt={file.name} />
            ) : (
              <Icon type="file" className={styles.fileIcon} />
            )}
          </div>
          <div className={styles.fileInfo}>
            <div className={styles.fileName} title={file.name}>{file.name}</div>
            <div className={styles.fileSize}>{(file.size / 1024).toFixed(2)} KB</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function KnowledgeBase() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadedDocs, setUploadedDocs] = useState([]);

  const onDrop = useCallback(acceptedFiles => {
    setFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleUpload = async () => {
    if (files.length === 0) return;
    
    setUploading(true);
    try {
      const response = await uploadFiles(files);
      setUploadedDocs(prev => [...prev, ...files.map(f => f.name)]);
      setFiles([]); // Clear files after successful upload
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className={styles.knowledge}>
      <div {...getRootProps()} className={classnames(styles.dropzone, isDragActive && styles.active)}>
        <input {...getInputProps()} />
        <Icon type="file" className={styles.uploadIcon} />
        <p>Drag & drop files here, or click to select files</p>
      </div>
      <FileList files={files} />
      {files.length > 0 && (
        <Button 
          type="primary" 
          onClick={handleUpload} 
          disabled={uploading}
          block
        >
          {uploading ? 'Uploading...' : 'Upload Files'}
        </Button>
      )}
      
      {uploadedDocs.length > 0 && (
        <div className={styles.uploadedDocs}>
          <Title type="h3">Added Documents:</Title>
          <ul className={styles.docsList}>
            {uploadedDocs.map((doc, index) => (
              <li key={index} className={styles.docItem}>
                <Icon type="file" className={styles.docIcon} />
                <span>{doc}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}