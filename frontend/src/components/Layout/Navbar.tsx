"use client"

import { useRouter } from 'next/navigation';

const Navbar = () => {
  const router = useRouter();

  return (
    <nav className="border-b border-neutral-800">
      <div className="mx-auto px-5">
        <div className="relative flex h-16 items-center justify-between">
          <div className="flex flex-1 sm:items-stretch sm:justify-start">
            <div className="flex flex-shrink-0 items-center">
              <div>
                <button type="button" onClick={() => router.push('/')}>
                  <h1 className='text-2xl font-semibold text-neutral-white'>健身肢體辨識系統</h1>
                </button>

                <p className="text-neutral-400">
                  一個基於 BiLSTM 和 MediaPipe 的健身肢體辨識系統，自動識別五種不同的運動模式。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar;
