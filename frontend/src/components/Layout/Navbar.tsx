"use client"

import { useRouter } from 'next/navigation';

const Navbar = () => {
  const router = useRouter();

  return (
    <nav className="border-b border-neutral-800">
      <div className="mx-auto px-5 py-3">
        <div className="relative flex items-center justify-between">
          <div className="flex flex-1 sm:items-stretch sm:justify-start">
            <div className="flex flex-shrink-0 items-center">
              <button type="button" onClick={() => router.push('/')}>
                <h1 className='text-2xl font-semibold text-neutral-white'>健身肢體辨識系統</h1>
              </button>
            </div>
          </div>
          <div className="flex items-center">
            {/* <button
              type="button"
              onClick={() => router.push('/docs')}
              className="text-neutral-300 hover:text-neutral-white transition-colors"
            >
              文檔
            </button> */}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar;
