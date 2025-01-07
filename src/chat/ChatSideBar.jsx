import React from 'react'
import { Avatar, Icon } from '../components'
import { useGlobal } from './context'
import styles from './style/sider.module'
import { classnames } from '../components/utils'

export function ChatSideBar() {
  const { is, setState } = useGlobal()
  return (
    <div className={classnames(styles.sider, 'flex-c-sb flex-column')}>
      <Avatar />
      <div className={classnames(styles.tool, 'flex-c-sb flex-column')}>
        <Icon className={styles.icon} type="history" onClick={() => setState({ is: { ...is, apps: false, knowledge: false } })} />
        <Icon className={styles.icon} type="file" onClick={() => setState({ is: { ...is, apps: false, knowledge: true } })} />
        <Icon className={styles.icon} type="config" onClick={() => setState({ is: { ...is, config: !is.config } })} />
        <Icon className={styles.icon} type={`${is.fullScreen ? 'min' : 'full'}-screen`} onClick={() => setState({ is: { ...is, fullScreen: !is.fullScreen } })} />
      </div>
    </div>
  )
}