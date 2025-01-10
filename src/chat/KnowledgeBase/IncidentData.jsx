import React, { useState, useEffect } from 'react';
import { getIncidents } from '../service/api';
import { Loading } from '@/components';
import styles from './knowledge.module.less';

export function IncidentData() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchIncidents() {
      try {
        const data = await getIncidents();
        setIncidents(data);
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
              <div className={styles.incidentText}>{incident.content}</div>
              {incident.metadata && (
                <div className={styles.incidentMeta}>
                  File: {incident.metadata.filename}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}