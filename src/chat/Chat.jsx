import React from 'react'
import { ChatMessage } from './ChatMessage'
import { ChatSideBar } from './ChatSideBar'
import { ChatOpitons } from './ChatOpitons'
import { ChatList } from './ChatList'
import { KnowledgeBase } from './KnowledgeBase'
import { classnames } from '../components/utils'
import { useGlobal } from './context'
import { Search } from '@/components'
import styles from './style/chat.module.less'
import { ScrollView } from './component'
import './style.less'

export default function Chat() {
  const { is } = useGlobal()
  const chatStyle = is.fullScreen ? styles.full : styles.normal
  const onSearch = (e) => {
    console.log(e)
  }
  return (
    <div className={classnames(styles.chat, chatStyle)}>
      <div className={styles.chat_inner}>
        <ChatSideBar />
        {
          is.config ?
            <ChatOpitons /> :
            <React.Fragment>
              {
                is.sidebar && <div className={styles.sider}>
                  {!is.knowledge ? (
                    <>
                      <div className={styles.search}>
                        <Search onSearch={onSearch} />
                      </div>
                      <ScrollView>
                        <ChatList />
                      </ScrollView>
                    </>
                  ) : (
                    <KnowledgeBase />
                  )}
                </div>
              }
              <ChatMessage />
            </React.Fragment>
        }
      </div>
    </div>
  )
}