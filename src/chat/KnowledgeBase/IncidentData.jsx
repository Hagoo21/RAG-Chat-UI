import React, { useState, useEffect } from 'react';
import { getIncidents } from '../service/api';
import { Loading } from '@/components';
import styles from './knowledge.module.less';

export function IncidentData() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [documentCounts, setDocumentCounts] = useState({});

  useEffect(() => {
    async function fetchIncidents() {
      try {
        const data = await getIncidents();
        
        // Get unique files only (first occurrence of each filename)
        const uniqueFiles = data.reduce((acc, current) => {
          const filename = current.metadata?.filename;
          if (filename && !acc.some(item => item.metadata?.filename === filename)) {
            acc.push(current);
          }
          return acc;
        }, []);
        
        setIncidents(uniqueFiles);
        
        // Calculate document counts per file
        const counts = data.reduce((acc, doc) => {
          const filename = doc.metadata?.filename;
          if (filename) {
            acc[filename] = (acc[filename] || 0) + 1;
          }
          return acc;
        }, {});
        setDocumentCounts(counts);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchIncidents();
  }, []);

  if (loading) return <Loading />;
  if (error) return <div className={styles.error}>Error loading incidents: {error}</div>;

  return (
    <div className={styles.incidents}>
      <h2 className={styles.incidentsTitle}>Incident Data</h2>
      <div className={styles.incidentsList}>
        {incidents.map((incident, index) => (
          <div key={index} className={styles.incidentItem}>
            <div className={styles.incidentContent}>
              {incident.metadata?.preview_image && (
                <div className={styles.incidentFilePreview}>
                  <img 
                    src={`data:image/png;base64,${incident.metadata.preview_image}`}
                    alt={`Preview of ${incident.metadata.filename}`}
                  />
                </div>
              )}
            </div>
            <div className={styles.fileInfo}>
              <div className={styles.fileName}>
                {incident.metadata?.filename}
              </div>
              <div className={styles.documentCount}>
                {documentCounts[incident.metadata?.filename]} embeddings
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
);

}